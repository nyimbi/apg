# Geographical Location Services - Technical Guide

**Version:** 2.0.0  
**Author:** Nyimbi Odero <nyimbi@gmail.com>  
**Company:** Datacraft  
**Website:** www.datacraft.co.ke  

## 📋 Table of Contents

1. [Technical Architecture](#technical-architecture)
2. [H3 Hierarchical Spatial Indexing](#h3-hierarchical-spatial-indexing)
3. [Fuzzy String Matching Algorithms](#fuzzy-string-matching-algorithms)
4. [Machine Learning Models](#machine-learning-models)
5. [Statistical Analysis Methods](#statistical-analysis-methods)
6. [Database Schema & Optimization](#database-schema--optimization)
7. [Real-Time Processing Pipeline](#real-time-processing-pipeline)
8. [Performance Optimization](#performance-optimization)
9. [Scalability Patterns](#scalability-patterns)
10. [Security Implementation](#security-implementation)
11. [Testing Strategies](#testing-strategies)
12. [Monitoring & Observability](#monitoring--observability)

---

## 🏗️ Technical Architecture

### System Overview

The Geographical Location Services capability is built on a microservices architecture with event-driven processing and multi-layered caching for optimal performance.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Load Balancer  │    │  Rate Limiter   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                          FastAPI Application                      │
├─────────────────┬───────────────┬───────────────┬─────────────────┤
│  Geocoding API  │ Trajectory    │ Hotspot       │ Predictive      │
│  H3 Indexing    │ Analysis      │ Detection     │ Modeling        │
│  Fuzzy Search   │ Pattern Recog │ Clustering    │ LSTM/ML         │
└─────────────────┴───────────────┴───────────────┴─────────────────┘
          │                      │                      │
┌─────────┼──────────────────────┼──────────────────────┼─────────┐
│         │           Service Layer (Async)            │         │
├─────────┼──────────────────────┼──────────────────────┼─────────┤
│ GLS     │ Fuzzy Matching    │ Trajectory Analysis   │ Hotspot │
│ Service │ Service           │ Service               │ Service │
└─────────┼──────────────────────┼──────────────────────┼─────────┘
          │                      │                      │
┌─────────┼──────────────────────┼──────────────────────┼─────────┐
│         │              Data Layer                    │         │
├─────────┼──────────────────────┼──────────────────────┼─────────┤
│ PostGIS │ Redis Cache       │ ML Model Store       │ Message │
│ Database│ (H3 Indices)      │ (TensorFlow/PyTorch) │ Queue   │
└─────────┴──────────────────────┴──────────────────────┴─────────┘
```

### Core Components

#### 1. API Layer (FastAPI)
```python
# FastAPI application with async support
app = FastAPI(
    title="Geographical Location Services",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware stack
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(LoggingMiddleware)
```

#### 2. Service Layer (Business Logic)
```python
class GeographicalLocationService:
    """Main service orchestrating all GLS operations."""
    
    def __init__(self):
        self.geocoder = GeocodingProvider()
        self.h3_indexer = H3SpatialIndexer()
        self.ml_models = MLModelManager()
        self.cache = CacheManager()
        self.db = DatabaseManager()
```

#### 3. Data Layer (PostgreSQL + PostGIS)
```sql
-- Core tables with spatial indexing
CREATE TABLE coordinates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(255) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    h3_indices JSONB NOT NULL DEFAULT '{}',
    geom GEOMETRY(POINT, 4326) GENERATED ALWAYS AS (ST_Point(longitude, latitude)) STORED,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Spatial index on geometry
CREATE INDEX idx_coordinates_geom ON coordinates USING GIST(geom);

-- H3 indices for fast spatial queries
CREATE INDEX idx_coordinates_h3_city ON coordinates USING BTREE((h3_indices->>'4'));
CREATE INDEX idx_coordinates_h3_district ON coordinates USING BTREE((h3_indices->>'5'));
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI + Uvicorn | High-performance async API |
| **Database** | PostgreSQL 15 + PostGIS 3.3 | Spatial data storage |
| **Caching** | Redis 7.0 | Multi-level caching |
| **Message Queue** | Redis Streams | Real-time event processing |
| **ML Framework** | TensorFlow 2.13 + PyTorch | Machine learning models |
| **Spatial Processing** | Shapely + GeoPandas | Geometric operations |
| **Monitoring** | Prometheus + Grafana | Metrics collection |
| **Logging** | Structured logging + ELK | Centralized logging |

---

## 🔷 H3 Hierarchical Spatial Indexing

### H3 Index Structure

H3 uses a hierarchical system of hexagonal cells to tile the Earth's surface. Each H3 index is a 64-bit integer that encodes:

- **Mode bits** (4 bits): Index type and version
- **Resolution bits** (4 bits): Resolution level (0-15)
- **Base cell bits** (7 bits): One of 122 base cells
- **Direction bits** (49 bits): Path through hierarchy

### Implementation Details

#### H3 Index Generation
```python
import struct
import hashlib

def generate_h3_indices(latitude: float, longitude: float) -> Dict[int, str]:
    """
    Generate H3 indices for all resolution levels.
    
    Note: This is a simplified implementation for demonstration.
    Production systems should use the official H3 library.
    """
    h3_indices = {}
    
    # Create a deterministic seed from coordinates
    coord_str = f"{latitude:.6f},{longitude:.6f}"
    seed = int(hashlib.md5(coord_str.encode()).hexdigest()[:8], 16)
    
    for resolution in range(11):  # Resolutions 0-10
        # Generate deterministic H3-like index
        resolution_seed = seed >> (resolution * 2)
        
        # H3 index format: mode(4) + res(4) + base_cell(7) + digits(49)
        mode = 1  # H3 cell index
        base_cell = resolution_seed % 122  # 122 base cells in H3
        digits = resolution_seed % (7 ** resolution) if resolution > 0 else 0
        
        # Pack into H3-like format
        h3_index = (mode << 60) | (resolution << 56) | (base_cell << 49) | digits
        
        # Convert to hex string format
        h3_indices[resolution] = f"8{resolution:01x}{h3_index & 0xffffffffffff:013x}"
    
    return h3_indices

def calculate_geohash(latitude: float, longitude: float, precision: int = 8) -> str:
    """Calculate geohash for coordinate with specified precision."""
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    
    lat_range = [-90.0, 90.0]
    lng_range = [-180.0, 180.0]
    
    geohash = []
    bits = 0
    bit_count = 0
    even_bit = True
    
    while len(geohash) < precision:
        if even_bit:  # Longitude
            mid = (lng_range[0] + lng_range[1]) / 2
            if longitude >= mid:
                bits = (bits << 1) | 1
                lng_range[0] = mid
            else:
                bits = bits << 1
                lng_range[1] = mid
        else:  # Latitude
            mid = (lat_range[0] + lat_range[1]) / 2
            if latitude >= mid:
                bits = (bits << 1) | 1
                lat_range[0] = mid
            else:
                bits = bits << 1
                lat_range[1] = mid
        
        even_bit = not even_bit
        bit_count += 1
        
        if bit_count == 5:
            geohash.append(base32[bits])
            bits = 0
            bit_count = 0
    
    return ''.join(geohash)
```

#### H3 Spatial Operations
```python
class H3SpatialIndexer:
    """H3 spatial indexing operations."""
    
    def get_neighbors(self, h3_index: str) -> List[str]:
        """Get neighboring H3 cells."""
        # Extract resolution and base components
        resolution = int(h3_index[1], 16)
        base_hash = int(h3_index[2:], 16)
        
        neighbors = []
        # Generate 6 hexagonal neighbors (simplified)
        for direction in range(6):
            neighbor_hash = base_hash ^ (1 << direction)
            neighbor_index = f"8{resolution:01x}{neighbor_hash:013x}"
            neighbors.append(neighbor_index)
        
        return neighbors
    
    def get_children(self, h3_index: str) -> List[str]:
        """Get child cells at next resolution level."""
        resolution = int(h3_index[1], 16)
        if resolution >= 10:  # Max resolution
            return []
        
        child_resolution = resolution + 1
        base_hash = int(h3_index[2:], 16)
        
        children = []
        # Each cell has 7 children in next resolution
        for child_idx in range(7):
            child_hash = (base_hash << 3) | child_idx
            child_index = f"8{child_resolution:01x}{child_hash:013x}"
            children.append(child_index)
        
        return children
    
    def get_resolution(self, h3_index: str) -> int:
        """Extract resolution level from H3 index."""
        return int(h3_index[1], 16)
    
    def h3_to_boundary(self, h3_index: str) -> List[Tuple[float, float]]:
        """Get hexagonal boundary coordinates for H3 cell."""
        # This is a simplified approximation
        resolution = self.get_resolution(h3_index)
        edge_length_km = [
            4250, 607, 86.7, 12.4, 1.77, 0.253, 0.0362, 0.00517, 
            0.000740, 0.000106, 0.0000151
        ][resolution]
        
        # Generate hexagonal vertices (simplified)
        center_lat, center_lng = self.h3_to_geo(h3_index)
        vertices = []
        
        for i in range(6):
            angle = i * 60 * (3.14159 / 180)  # 60 degrees in radians
            lat_offset = edge_length_km * 0.001 * math.cos(angle)
            lng_offset = edge_length_km * 0.001 * math.sin(angle)
            
            vertices.append((
                center_lat + lat_offset,
                center_lng + lng_offset
            ))
        
        return vertices
```

### H3 Query Optimization

#### Spatial Range Queries
```python
async def find_entities_in_h3_range(
    self, 
    center_h3: str, 
    ring_size: int = 1
) -> List[GLSCoordinate]:
    """Find all entities within H3 ring distance."""
    
    # Get all H3 cells within range
    target_cells = set([center_h3])
    
    for ring in range(1, ring_size + 1):
        ring_cells = self.get_h3_ring(center_h3, ring)
        target_cells.update(ring_cells)
    
    # Query database using H3 index
    resolution = self.get_resolution(center_h3)
    query = """
        SELECT entity_id, latitude, longitude, h3_indices, timestamp
        FROM coordinates 
        WHERE h3_indices->>%s = ANY(%s)
        ORDER BY timestamp DESC
    """
    
    results = await self.db.fetch_all(
        query, 
        str(resolution), 
        list(target_cells)
    )
    
    return [GLSCoordinate.from_db_row(row) for row in results]

def get_h3_ring(self, center_h3: str, ring_distance: int) -> Set[str]:
    """Get H3 cells at specified ring distance."""
    if ring_distance == 0:
        return {center_h3}
    
    # Start with center cell
    current_ring = {center_h3}
    
    for distance in range(ring_distance):
        next_ring = set()
        for cell in current_ring:
            # Add neighbors of each cell in current ring
            neighbors = self.get_neighbors(cell)
            next_ring.update(neighbors)
        current_ring = next_ring
    
    return current_ring
```

#### Temporal-Spatial Indexing
```python
class TemporalSpatialIndex:
    """Combined temporal and spatial indexing for trajectory data."""
    
    def __init__(self, h3_resolution: int = 6):
        self.h3_resolution = h3_resolution
        self.cache = {}
    
    def create_spatiotemporal_key(
        self, 
        h3_index: str, 
        timestamp: datetime, 
        time_bucket_minutes: int = 15
    ) -> str:
        """Create compound key for spatiotemporal indexing."""
        
        # Truncate timestamp to time bucket
        time_bucket = timestamp.replace(
            minute=(timestamp.minute // time_bucket_minutes) * time_bucket_minutes,
            second=0,
            microsecond=0
        )
        
        # Combine H3 index with time bucket
        time_str = time_bucket.strftime("%Y%m%d_%H%M")
        return f"{h3_index}_{time_str}"
    
    async def query_spatiotemporal_range(
        self,
        h3_cells: List[str],
        start_time: datetime,
        end_time: datetime,
        time_bucket_minutes: int = 15
    ) -> List[GLSCoordinate]:
        """Query entities across spatial and temporal dimensions."""
        
        # Generate all spatiotemporal keys
        keys = []
        current_time = start_time
        
        while current_time <= end_time:
            for h3_cell in h3_cells:
                key = self.create_spatiotemporal_key(
                    h3_cell, current_time, time_bucket_minutes
                )
                keys.append(key)
            
            current_time += timedelta(minutes=time_bucket_minutes)
        
        # Batch query from cache/database
        return await self.batch_query_keys(keys)
```

---

## 🔤 Fuzzy String Matching Algorithms

### Algorithm Implementations

#### 1. Levenshtein Distance
```python
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate edit distance between two strings.
    Dynamic programming implementation with O(m*n) complexity.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # Create matrix
    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Convert Levenshtein distance to similarity score (0-1)."""
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1 - (distance / max_len) if max_len > 0 else 1.0
```

#### 2. Jaro-Winkler Similarity
```python
def jaro_similarity(s1: str, s2: str) -> float:
    """Calculate Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Calculate matching window
    match_window = max(len1, len2) // 2 - 1
    match_window = max(0, match_window)
    
    # Initialize match arrays
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        
        while not s2_matches[k]:
            k += 1
        
        if s1[i] != s2[k]:
            transpositions += 1
        
        k += 1
    
    # Calculate Jaro similarity
    jaro = (matches / len1 + matches / len2 + 
            (matches - transpositions / 2) / matches) / 3
    
    return jaro

def jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1) -> float:
    """Calculate Jaro-Winkler similarity with prefix scaling."""
    jaro_sim = jaro_similarity(s1.lower(), s2.lower())
    
    if jaro_sim < 0.7:  # Only apply prefix scaling if Jaro > 0.7
        return jaro_sim
    
    # Calculate common prefix length (up to 4 characters)
    prefix_length = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i].lower() == s2[i].lower():
            prefix_length += 1
        else:
            break
    
    # Apply prefix scaling
    return jaro_sim + (prefix_length * prefix_scale * (1 - jaro_sim))
```

#### 3. Soundex Algorithm
```python
def soundex(name: str) -> str:
    """
    Generate Soundex code for phonetic matching.
    Returns 4-character code: first letter + 3 digits.
    """
    if not name:
        return "0000"
    
    name = name.upper().strip()
    
    # Soundex mapping
    soundex_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    # Keep first letter
    result = name[0]
    
    # Process remaining characters
    for i in range(1, len(name)):
        char = name[i]
        
        # Skip vowels and some consonants
        if char in 'AEIOUHWY':
            continue
        
        # Get soundex digit
        digit = soundex_map.get(char, '')
        
        if digit:
            # Avoid consecutive duplicates
            if not result[-1:].isdigit() or result[-1] != digit:
                result += digit
    
    # Pad or truncate to 4 characters
    result = (result + '000')[:4]
    
    return result

def soundex_similarity(s1: str, s2: str) -> float:
    """Calculate similarity based on Soundex codes."""
    soundex1 = soundex(s1)
    soundex2 = soundex(s2)
    
    if soundex1 == soundex2:
        return 1.0
    
    # Partial matching based on character positions
    matches = sum(1 for c1, c2 in zip(soundex1, soundex2) if c1 == c2)
    return matches / 4.0
```

#### 4. Metaphone Algorithm (Simplified)
```python
def metaphone(word: str, max_length: int = 4) -> str:
    """
    Generate Metaphone code for advanced phonetic matching.
    Simplified version of the Double Metaphone algorithm.
    """
    if not word:
        return ""
    
    word = word.upper().strip()
    result = ""
    i = 0
    
    # Skip initial silent letters
    if word.startswith(('KN', 'GN', 'PN', 'AE', 'WR')):
        i = 1
    elif word.startswith('X'):
        result = 'S'
        i = 1
    
    while i < len(word) and len(result) < max_length:
        char = word[i]
        
        # Character-specific rules
        if char == 'B':
            if i == len(word) - 1 and word[i-1:i+1] == 'MB':
                pass  # Silent B in MB
            else:
                result += 'B'
        
        elif char == 'C':
            if i > 0 and word[i-1] == 'S' and char in 'EIY':
                pass  # SCE, SCI, SCY
            elif word[i:i+2] == 'CH':
                result += 'X'
                i += 1
            elif word[i:i+2] in ('CE', 'CI'):
                result += 'S'
            else:
                result += 'K'
        
        elif char == 'D':
            if word[i:i+2] == 'DG' and word[i+2:i+3] in 'EIY':
                result += 'J'
                i += 2
            else:
                result += 'T'
        
        elif char == 'G':
            if word[i:i+2] == 'GH':
                if i == 0 or word[i-1] in 'AEIOU':
                    result += 'G'
                i += 1
            elif word[i:i+2] == 'GN':
                result += 'N'
                i += 1
            elif word[i+1:i+2] in 'EIY':
                result += 'J'
            else:
                result += 'K'
        
        elif char == 'H':
            if i == 0 or word[i-1] in 'AEIOU':
                if word[i+1:i+2] in 'AEIOU':
                    result += 'H'
        
        elif char in 'AEIOU':
            if i == 0:
                result += char
        
        elif char == 'J':
            result += 'J'
        
        elif char == 'K':
            if i == 0 or word[i-1] != 'C':
                result += 'K'
        
        elif char == 'L':
            result += 'L'
        
        elif char == 'M':
            result += 'M'
        
        elif char == 'N':
            result += 'N'
        
        elif char == 'P':
            if word[i:i+2] == 'PH':
                result += 'F'
                i += 1
            else:
                result += 'P'
        
        elif char == 'Q':
            result += 'K'
        
        elif char == 'R':
            result += 'R'
        
        elif char == 'S':
            if word[i:i+2] == 'SH':
                result += 'X'
                i += 1
            else:
                result += 'S'
        
        elif char == 'T':
            if word[i:i+2] == 'TH':
                result += '0'  # TH sound
                i += 1
            elif word[i:i+3] in ('TIA', 'TIO'):
                result += 'X'
                i += 2
            else:
                result += 'T'
        
        elif char == 'V':
            result += 'F'
        
        elif char == 'W':
            if word[i+1:i+2] in 'AEIOU':
                result += 'W'
        
        elif char == 'X':
            result += 'KS'
        
        elif char == 'Y':
            if word[i+1:i+2] in 'AEIOU':
                result += 'Y'
        
        elif char == 'Z':
            result += 'S'
        
        i += 1
    
    return result[:max_length]

def metaphone_similarity(s1: str, s2: str) -> float:
    """Calculate similarity based on Metaphone codes."""
    meta1 = metaphone(s1)
    meta2 = metaphone(s2)
    
    if meta1 == meta2:
        return 1.0
    
    # Use Levenshtein on metaphone codes
    return levenshtein_similarity(meta1, meta2)
```

### Fuzzy Matching Service Implementation

```python
class GLSFuzzyMatchingService:
    """Advanced fuzzy matching service for geographic names."""
    
    def __init__(self):
        self.geonames_index = self._load_geonames_index()
        self.admin_hierarchy = self._load_admin_hierarchy()
        self.algorithm_map = {
            GLSFuzzyMatchType.LEVENSHTEIN: levenshtein_similarity,
            GLSFuzzyMatchType.JARO_WINKLER: jaro_winkler_similarity,
            GLSFuzzyMatchType.SOUNDEX: soundex_similarity,
            GLSFuzzyMatchType.METAPHONE: metaphone_similarity,
            GLSFuzzyMatchType.FUZZY_WUZZY: self._fuzzy_wuzzy_similarity
        }
    
    async def fuzzy_search(
        self, 
        request: GLSFuzzySearchRequest
    ) -> List[GLSFuzzyMatch]:
        """Perform fuzzy search with specified algorithm and filters."""
        
        # Get similarity function
        similarity_func = self.algorithm_map[request.fuzzy_match_type]
        
        # Apply filters
        candidates = self._filter_candidates(request)
        
        # Calculate similarities
        matches = []
        for candidate in candidates:
            similarity = similarity_func(request.query_text, candidate.name)
            
            if similarity >= request.confidence_threshold:
                match = GLSFuzzyMatch(
                    name=candidate.name,
                    coordinate=candidate.coordinate,
                    confidence_score=similarity,
                    admin_hierarchy=candidate.admin_hierarchy,
                    data_source="geonames",
                    feature_code=candidate.feature_code
                )
                matches.append(match)
        
        # Sort by confidence and return top results
        matches.sort(key=lambda x: x.confidence_score, reverse=True)
        return matches[:request.max_results]
    
    def _filter_candidates(
        self, 
        request: GLSFuzzySearchRequest
    ) -> List[GeographicFeature]:
        """Apply administrative and geographic filters."""
        candidates = self.geonames_index
        
        # Country filter
        if request.country_filter:
            candidates = [
                c for c in candidates 
                if c.admin_hierarchy.country_code == request.country_filter
            ]
        
        # Administrative level filter
        if request.admin_level:
            candidates = [
                c for c in candidates
                if c.feature_type == request.admin_level.value
            ]
        
        return candidates
    
    def _fuzzy_wuzzy_similarity(self, s1: str, s2: str) -> float:
        """FuzzyWuzzy-style similarity calculation."""
        # Token set ratio
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if not union:
            return 1.0 if not tokens1 and not tokens2 else 0.0
        
        token_set_ratio = len(intersection) / len(union)
        
        # Combine with Jaro-Winkler for final score
        jw_ratio = jaro_winkler_similarity(s1, s2)
        
        return max(token_set_ratio, jw_ratio)
```

---

## 🤖 Machine Learning Models

### LSTM Trajectory Prediction

#### Model Architecture
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class TrajectoryLSTMModel:
    """LSTM model for trajectory prediction."""
    
    def __init__(self, 
                 sequence_length: int = 50,
                 feature_dim: int = 6,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.2):
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """Build LSTM model architecture."""
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # First LSTM layer with return sequences
        x = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        )(inputs)
        
        # Second LSTM layer
        x = layers.LSTM(
            self.lstm_units // 2,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        )(x)
        
        # Dense layers for prediction
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer: [lat, lng, confidence]
        outputs = layers.Dense(3, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_features(self, coordinates: List[GLSCoordinate]) -> np.ndarray:
        """Prepare input features for LSTM."""
        features = []
        
        for i, coord in enumerate(coordinates):
            # Temporal features
            hour = coord.timestamp.hour / 24.0
            day_of_week = coord.timestamp.weekday() / 7.0
            
            # Spatial features
            lat_norm = (coord.latitude + 90) / 180  # Normalize to 0-1
            lng_norm = (coord.longitude + 180) / 360  # Normalize to 0-1
            
            # Movement features (if not first point)
            if i > 0:
                prev_coord = coordinates[i-1]
                speed = self._calculate_speed(prev_coord, coord)
                bearing = self._calculate_bearing(prev_coord, coord)
            else:
                speed = 0.0
                bearing = 0.0
            
            features.append([
                lat_norm, lng_norm, hour, day_of_week, 
                speed / 100.0, bearing / 360.0  # Normalize
            ])
        
        return np.array(features)
    
    def _calculate_speed(self, coord1: GLSCoordinate, coord2: GLSCoordinate) -> float:
        """Calculate speed between two coordinates in km/h."""
        distance_km = self._haversine_distance(coord1, coord2)
        time_diff_hours = (coord2.timestamp - coord1.timestamp).total_seconds() / 3600
        
        if time_diff_hours == 0:
            return 0.0
        
        return distance_km / time_diff_hours
    
    def _calculate_bearing(self, coord1: GLSCoordinate, coord2: GLSCoordinate) -> float:
        """Calculate bearing between two coordinates in degrees."""
        lat1, lng1 = math.radians(coord1.latitude), math.radians(coord1.longitude)
        lat2, lng2 = math.radians(coord2.latitude), math.radians(coord2.longitude)
        
        dlon = lng2 - lng1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    async def predict_trajectory(
        self, 
        historical_coords: List[GLSCoordinate],
        prediction_steps: int
    ) -> List[GLSPredictionModel]:
        """Predict future trajectory points."""
        
        # Prepare features
        features = self.prepare_features(historical_coords)
        
        if len(features) < self.sequence_length:
            # Pad with zeros if insufficient history
            padding = np.zeros((self.sequence_length - len(features), self.feature_dim))
            features = np.vstack([padding, features])
        
        # Use last sequence_length points for prediction
        input_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, self.feature_dim)
        
        predictions = []
        current_sequence = input_sequence.copy()
        
        for step in range(prediction_steps):
            # Predict next point
            pred = self.model.predict(current_sequence, verbose=0)
            
            # Extract prediction components
            lat_norm, lng_norm, confidence = pred[0]
            
            # Denormalize coordinates
            predicted_lat = lat_norm * 180 - 90
            predicted_lng = lng_norm * 360 - 180
            
            # Create prediction object
            prediction_time = historical_coords[-1].timestamp + timedelta(hours=step+1)
            predicted_coord = GLSCoordinate(
                latitude=float(predicted_lat),
                longitude=float(predicted_lng)
            )
            
            prediction = GLSPredictionModel(
                entity_id=historical_coords[0].entity_id,
                predicted_coordinate=predicted_coord,
                prediction_timestamp=prediction_time,
                confidence_score=float(max(0.0, min(1.0, confidence))),
                model_type="lstm",
                prediction_horizon_hours=step + 1,
                risk_score=self._calculate_risk_score(predicted_coord, historical_coords),
                risk_factors=self._identify_risk_factors(predicted_coord)
            )
            predictions.append(prediction)
            
            # Update sequence for next prediction
            next_features = self.prepare_features([predicted_coord])[-1:]
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features[0]
        
        return predictions
```

### Anomaly Detection Models

#### Isolation Forest for Spatial Anomalies
```python
from sklearn.ensemble import IsolationForest
import numpy as np

class SpatialAnomalyDetector:
    """Isolation Forest-based spatial anomaly detection."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False
    
    def fit(self, coordinates: List[GLSCoordinate]):
        """Train anomaly detection model on normal coordinate patterns."""
        features = self._extract_spatial_features(coordinates)
        self.model.fit(features)
        self.is_fitted = True
    
    def detect_anomalies(
        self, 
        coordinates: List[GLSCoordinate]
    ) -> List[GLSAnomalyDetection]:
        """Detect spatial anomalies in coordinate sequence."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        features = self._extract_spatial_features(coordinates)
        anomaly_scores = self.model.decision_function(features)
        anomaly_labels = self.model.predict(features)
        
        anomalies = []
        for i, (coord, score, label) in enumerate(zip(coordinates, anomaly_scores, anomaly_labels)):
            if label == -1:  # Anomaly detected
                # Convert score to 0-1 range (higher = more anomalous)
                normalized_score = (1 - score) / 2
                
                anomaly = GLSAnomalyDetection(
                    anomaly_id=f"spatial_anom_{i}",
                    entity_id=coord.entity_id,
                    anomaly_type="spatial",
                    timestamp=coord.timestamp,
                    coordinate=coord,
                    anomaly_score=float(max(0.0, min(1.0, normalized_score))),
                    severity_level=self._classify_severity(normalized_score),
                    description=f"Spatial location significantly deviates from normal patterns",
                    context={
                        "isolation_score": float(score),
                        "feature_importance": self._get_feature_importance(features[i])
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _extract_spatial_features(self, coordinates: List[GLSCoordinate]) -> np.ndarray:
        """Extract spatial features for anomaly detection."""
        features = []
        
        for i, coord in enumerate(coordinates):
            # Basic spatial features
            spatial_features = [coord.latitude, coord.longitude]
            
            # Distance from centroid
            if len(coordinates) > 1:
                centroid_lat = np.mean([c.latitude for c in coordinates])
                centroid_lng = np.mean([c.longitude for c in coordinates])
                centroid_distance = self._haversine_distance(
                    coord, GLSCoordinate(latitude=centroid_lat, longitude=centroid_lng)
                )
                spatial_features.append(centroid_distance)
            else:
                spatial_features.append(0.0)
            
            # Movement features (if not first point)
            if i > 0:
                prev_coord = coordinates[i-1]
                distance = self._haversine_distance(prev_coord, coord)
                bearing = self._calculate_bearing(prev_coord, coord)
                spatial_features.extend([distance, bearing])
            else:
                spatial_features.extend([0.0, 0.0])
            
            # Temporal features
            hour_sin = np.sin(2 * np.pi * coord.timestamp.hour / 24)
            hour_cos = np.cos(2 * np.pi * coord.timestamp.hour / 24)
            dow_sin = np.sin(2 * np.pi * coord.timestamp.weekday() / 7)
            dow_cos = np.cos(2 * np.pi * coord.timestamp.weekday() / 7)
            spatial_features.extend([hour_sin, hour_cos, dow_sin, dow_cos])
            
            features.append(spatial_features)
        
        return np.array(features)
```

### Clustering Algorithms for Hotspot Detection

#### DBSCAN Implementation
```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class SpatialClusterer:
    """Spatial clustering for hotspot detection."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.dbscan = None
    
    async def detect_hotspots_dbscan(
        self,
        coordinates: List[GLSCoordinate],
        eps: float = 0.5,  # Maximum distance between points in cluster
        min_samples: int = 5  # Minimum points to form cluster
    ) -> List[GLSHotspot]:
        """Detect hotspots using DBSCAN clustering."""
        
        if len(coordinates) < min_samples:
            return []
        
        # Extract coordinate pairs
        points = np.array([[coord.latitude, coord.longitude] for coord in coordinates])
        
        # Scale coordinates for clustering
        points_scaled = self.scaler.fit_transform(points)
        
        # Perform DBSCAN clustering
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan.fit_predict(points_scaled)
        
        # Process clusters
        hotspots = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get cluster points
            cluster_mask = cluster_labels == label
            cluster_coords = [coord for i, coord in enumerate(coordinates) if cluster_mask[i]]
            cluster_points = points[cluster_mask]
            
            # Calculate cluster center
            center_lat = np.mean(cluster_points[:, 0])
            center_lng = np.mean(cluster_points[:, 1])
            center_coord = GLSCoordinate(latitude=center_lat, longitude=center_lng)
            
            # Calculate cluster statistics
            intensity_score = len(cluster_coords) / len(coordinates)
            
            # Statistical significance using z-score
            expected_density = len(coordinates) / self._calculate_area(coordinates)
            cluster_density = len(cluster_coords) / self._calculate_cluster_area(cluster_points)
            z_score = (cluster_density - expected_density) / np.std([cluster_density, expected_density])
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            hotspot = GLSHotspot(
                hotspot_id=f"dbscan_cluster_{label}",
                center_coordinate=center_coord,
                entity_count=len(cluster_coords),
                intensity_score=intensity_score,
                z_score=z_score,
                statistical_significance=p_value,
                time_window_start=min(coord.timestamp for coord in cluster_coords),
                time_window_end=max(coord.timestamp for coord in cluster_coords),
                h3_resolution=GLSH3Resolution.NEIGHBORHOOD,
                cluster_radius_meters=self._calculate_cluster_radius(cluster_points)
            )
            hotspots.append(hotspot)
        
        return hotspots
    
    def _calculate_cluster_radius(self, points: np.ndarray) -> float:
        """Calculate radius of cluster in meters."""
        if len(points) < 2:
            return 0.0
        
        center = np.mean(points, axis=0)
        distances = [
            self._haversine_distance(
                GLSCoordinate(latitude=center[0], longitude=center[1]),
                GLSCoordinate(latitude=point[0], longitude=point[1])
            ) * 1000  # Convert to meters
            for point in points
        ]
        
        return float(np.mean(distances))
```

---

## 📊 Statistical Analysis Methods

### Spatial Autocorrelation (Moran's I)

```python
import numpy as np
from scipy import stats
from typing import List, Tuple

class SpatialStatistics:
    """Statistical methods for spatial analysis."""
    
    def calculate_global_morans_i(
        self,
        coordinates: List[GLSCoordinate],
        values: List[float],
        distance_threshold: float = 1000  # meters
    ) -> Dict[str, float]:
        """
        Calculate Global Moran's I statistic for spatial autocorrelation.
        
        Returns:
            Dictionary with statistic, expected value, variance, z-score, p-value
        """
        n = len(coordinates)
        if n < 3:
            raise ValueError("Need at least 3 points for Moran's I calculation")
        
        # Create spatial weights matrix
        W = self._create_spatial_weights_matrix(coordinates, distance_threshold)
        
        # Calculate Moran's I
        values_array = np.array(values)
        mean_value = np.mean(values_array)
        
        # Deviations from mean
        z = values_array - mean_value
        
        # Numerator: sum of weighted cross-products
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += W[i][j] * z[i] * z[j]
        
        # Denominator: sum of squared deviations
        denominator = np.sum(z ** 2)
        
        # Sum of weights
        W_sum = np.sum(W)
        
        # Moran's I statistic
        if denominator == 0 or W_sum == 0:
            morans_i = 0.0
        else:
            morans_i = (n / W_sum) * (numerator / denominator)
        
        # Expected value under null hypothesis
        expected_i = -1.0 / (n - 1)
        
        # Variance calculation
        # This is a simplified version - full calculation is more complex
        variance_i = self._calculate_morans_i_variance(W, n)
        
        # Z-score and p-value
        if variance_i > 0:
            z_score = (morans_i - expected_i) / np.sqrt(variance_i)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = 0.0
            p_value = 1.0
        
        return {
            "statistic": morans_i,
            "expected_value": expected_i,
            "variance": variance_i,
            "z_score": z_score,
            "p_value": p_value,
            "interpretation": self._interpret_morans_i(morans_i, p_value)
        }
    
    def calculate_local_morans_i(
        self,
        coordinates: List[GLSCoordinate],
        values: List[float],
        distance_threshold: float = 1000
    ) -> List[Dict[str, float]]:
        """Calculate Local Moran's I (LISA) for each location."""
        n = len(coordinates)
        W = self._create_spatial_weights_matrix(coordinates, distance_threshold)
        
        values_array = np.array(values)
        mean_value = np.mean(values_array)
        std_value = np.std(values_array)
        
        # Standardized values
        z = (values_array - mean_value) / std_value if std_value > 0 else np.zeros(n)
        
        local_morans = []
        
        for i in range(n):
            # Local Moran's I for location i
            neighbors_sum = sum(W[i][j] * z[j] for j in range(n))
            local_i = z[i] * neighbors_sum
            
            # Calculate local variance (simplified)
            wi_sum = sum(W[i])
            local_variance = wi_sum * (1 - wi_sum / (n - 1))
            
            # Z-score and p-value
            if local_variance > 0:
                z_score = local_i / np.sqrt(local_variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0.0
                p_value = 1.0
            
            # Classify cluster type
            cluster_type = self._classify_lisa_cluster(z[i], neighbors_sum, p_value)
            
            local_morans.append({
                "local_i": local_i,
                "z_score": z_score,
                "p_value": p_value,
                "cluster_type": cluster_type,
                "coordinate": coordinates[i]
            })
        
        return local_morans
    
    def _create_spatial_weights_matrix(
        self,
        coordinates: List[GLSCoordinate],
        distance_threshold: float
    ) -> List[List[float]]:
        """Create spatial weights matrix based on distance threshold."""
        n = len(coordinates)
        W = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = self._haversine_distance(coordinates[i], coordinates[j]) * 1000  # meters
                    if distance <= distance_threshold:
                        # Inverse distance weighting
                        W[i][j] = 1.0 / max(distance, 1.0)
        
        # Row-standardize weights
        for i in range(n):
            row_sum = sum(W[i])
            if row_sum > 0:
                for j in range(n):
                    W[i][j] /= row_sum
        
        return W
    
    def _calculate_morans_i_variance(self, W: List[List[float]], n: int) -> float:
        """Calculate variance of Moran's I statistic."""
        # Simplified variance calculation
        W_sum = sum(sum(row) for row in W)
        
        if W_sum == 0:
            return 0.0
        
        # This is a simplified approximation
        # Full calculation involves S0, S1, S2 terms
        S0 = W_sum
        S1 = 0.5 * sum(sum((W[i][j] + W[j][i]) ** 2 for j in range(n)) for i in range(n))
        S2 = sum(sum(W[i][j] for j in range(n)) ** 2 for i in range(n))
        
        variance = ((n * S1 - 2 * S2) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)) - (1 / (n - 1)) ** 2
        
        return max(variance, 0.0)
    
    def _interpret_morans_i(self, morans_i: float, p_value: float) -> str:
        """Interpret Moran's I result."""
        if p_value > 0.05:
            return "no_significant_autocorrelation"
        elif morans_i > 0:
            return "significant_positive_autocorrelation"
        else:
            return "significant_negative_autocorrelation"
    
    def _classify_lisa_cluster(self, z_i: float, neighbors_sum: float, p_value: float) -> str:
        """Classify LISA cluster type."""
        if p_value > 0.05:
            return "not_significant"
        
        if z_i > 0 and neighbors_sum > 0:
            return "high_high"
        elif z_i < 0 and neighbors_sum < 0:
            return "low_low"
        elif z_i > 0 and neighbors_sum < 0:
            return "high_low"
        elif z_i < 0 and neighbors_sum > 0:
            return "low_high"
        else:
            return "not_significant"
```

### Getis-Ord Gi* Statistics

```python
class GetisOrdStatistics:
    """Getis-Ord Gi* statistics for hotspot analysis."""
    
    def calculate_getis_ord_gi_star(
        self,
        coordinates: List[GLSCoordinate],
        values: List[float],
        distance_threshold: float = 1000
    ) -> List[Dict[str, float]]:
        """
        Calculate Getis-Ord Gi* statistic for hotspot detection.
        
        Returns list of statistics for each location.
        """
        n = len(coordinates)
        values_array = np.array(values)
        
        # Global statistics
        global_mean = np.mean(values_array)
        global_std = np.std(values_array)
        
        gi_star_results = []
        
        for i in range(n):
            # Create neighborhood (including location i)
            neighbors = [i]  # Include self
            for j in range(n):
                if i != j:
                    distance = self._haversine_distance(coordinates[i], coordinates[j]) * 1000
                    if distance <= distance_threshold:
                        neighbors.append(j)
            
            # Calculate Gi* statistic
            W_i = len(neighbors)  # Number of neighbors (including self)
            sum_w_x = sum(values_array[j] for j in neighbors)
            
            # Expected value and variance
            expected_gi = W_i * global_mean
            
            if n > 1 and global_std > 0:
                variance_gi = (W_i * (n - W_i) * global_std ** 2) / (n - 1)
                
                if variance_gi > 0:
                    gi_star = (sum_w_x - expected_gi) / np.sqrt(variance_gi)
                    p_value = 2 * (1 - stats.norm.cdf(abs(gi_star)))
                else:
                    gi_star = 0.0
                    p_value = 1.0
            else:
                gi_star = 0.0
                p_value = 1.0
            
            # Classify hotspot type
            hotspot_type = self._classify_getis_ord_hotspot(gi_star, p_value)
            
            gi_star_results.append({
                "gi_star": gi_star,
                "p_value": p_value,
                "z_score": gi_star,  # Gi* is already a z-score
                "hotspot_type": hotspot_type,
                "neighbors_count": W_i,
                "coordinate": coordinates[i],
                "local_sum": sum_w_x,
                "expected_sum": expected_gi
            })
        
        return gi_star_results
    
    def _classify_getis_ord_hotspot(self, gi_star: float, p_value: float) -> str:
        """Classify Getis-Ord hotspot type."""
        if p_value > 0.05:
            return "not_significant"
        elif gi_star > 2.58:  # 99% confidence
            return "high_confidence_hotspot"
        elif gi_star > 1.96:  # 95% confidence
            return "hotspot"
        elif gi_star < -2.58:  # 99% confidence
            return "high_confidence_coldspot"
        elif gi_star < -1.96:  # 95% confidence
            return "coldspot"
        else:
            return "not_significant"
```

---

## 🗄️ Database Schema & Optimization

### PostGIS Schema Design

```sql
-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main coordinates table with spatial indexing
CREATE TABLE coordinates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude DECIMAL(11, 8) NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    altitude DECIMAL(8, 2) DEFAULT NULL,
    
    -- H3 indices for hierarchical spatial indexing
    h3_indices JSONB NOT NULL DEFAULT '{}',
    primary_h3_index VARCHAR(15) GENERATED ALWAYS AS (h3_indices->>'4') STORED,
    
    -- PostGIS geometry column
    geom GEOMETRY(POINT, 4326) GENERATED ALWAYS AS (ST_Point(longitude, latitude)) STORED,
    
    -- Geohash for additional spatial indexing
    geohash VARCHAR(12) DEFAULT NULL,
    
    -- Temporal indexing
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Metadata
    accuracy_meters DECIMAL(8, 2) DEFAULT NULL,
    speed_kmh DECIMAL(8, 2) DEFAULT NULL,
    bearing_degrees DECIMAL(5, 2) DEFAULT NULL,
    metadata JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(255) NOT NULL
);

-- Spatial indices
CREATE INDEX idx_coordinates_geom ON coordinates USING GIST(geom);
CREATE INDEX idx_coordinates_geom_time ON coordinates USING GIST(geom, timestamp);

-- H3 indices for fast spatial queries
CREATE INDEX idx_coordinates_h3_continent ON coordinates USING BTREE((h3_indices->>'0'));
CREATE INDEX idx_coordinates_h3_country ON coordinates USING BTREE((h3_indices->>'1'));
CREATE INDEX idx_coordinates_h3_state ON coordinates USING BTREE((h3_indices->>'2'));
CREATE INDEX idx_coordinates_h3_metro ON coordinates USING BTREE((h3_indices->>'3'));
CREATE INDEX idx_coordinates_h3_city ON coordinates USING BTREE(primary_h3_index);
CREATE INDEX idx_coordinates_h3_district ON coordinates USING BTREE((h3_indices->>'5'));
CREATE INDEX idx_coordinates_h3_neighborhood ON coordinates USING BTREE((h3_indices->>'6'));

-- Entity and temporal indices
CREATE INDEX idx_coordinates_entity_time ON coordinates(entity_id, timestamp DESC);
CREATE INDEX idx_coordinates_entity_type ON coordinates(entity_type);
CREATE INDEX idx_coordinates_timestamp ON coordinates(timestamp DESC);
CREATE INDEX idx_coordinates_tenant ON coordinates(tenant_id);

-- Geohash index
CREATE INDEX idx_coordinates_geohash ON coordinates(geohash) WHERE geohash IS NOT NULL;

-- Composite indices for common queries
CREATE INDEX idx_coordinates_entity_h3_time ON coordinates(entity_id, primary_h3_index, timestamp DESC);
CREATE INDEX idx_coordinates_type_h3 ON coordinates(entity_type, primary_h3_index);

-- Trajectories table for analyzed movement patterns
CREATE TABLE trajectories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(255) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    
    -- Pattern analysis results
    detected_patterns TEXT[] DEFAULT '{}',
    total_distance_meters DECIMAL(12, 2) NOT NULL,
    average_speed_kmh DECIMAL(8, 2) DEFAULT NULL,
    max_speed_kmh DECIMAL(8, 2) DEFAULT NULL,
    
    -- H3 analysis
    h3_resolution INTEGER NOT NULL,
    h3_cells_visited TEXT[] DEFAULT '{}',
    
    -- Dwell points
    dwell_points JSONB DEFAULT '[]',
    
    -- Anomaly scores
    anomaly_scores JSONB DEFAULT '[]',
    overall_anomaly_score DECIMAL(5, 4) DEFAULT NULL,
    
    -- Metadata
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(255) NOT NULL
);

-- Trajectory indices
CREATE INDEX idx_trajectories_entity_time ON trajectories(entity_id, start_time DESC);
CREATE INDEX idx_trajectories_time_range ON trajectories(start_time, end_time);
CREATE INDEX idx_trajectories_patterns ON trajectories USING GIN(detected_patterns);
CREATE INDEX idx_trajectories_h3_cells ON trajectories USING GIN(h3_cells_visited);

-- Hotspots table for detected spatial clusters
CREATE TABLE hotspots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotspot_id VARCHAR(255) NOT NULL,
    
    -- Spatial properties
    center_geom GEOMETRY(POINT, 4326) NOT NULL,
    center_h3_index VARCHAR(15) NOT NULL,
    cluster_radius_meters DECIMAL(8, 2) NOT NULL,
    
    -- Statistical properties
    entity_count INTEGER NOT NULL,
    intensity_score DECIMAL(5, 4) NOT NULL,
    z_score DECIMAL(8, 4) NOT NULL,
    statistical_significance DECIMAL(10, 8) NOT NULL,
    
    -- Temporal window
    time_window_start TIMESTAMPTZ NOT NULL,
    time_window_end TIMESTAMPTZ NOT NULL,
    
    -- Analysis parameters
    clustering_algorithm VARCHAR(50) NOT NULL,
    h3_resolution INTEGER NOT NULL,
    min_cluster_size INTEGER NOT NULL,
    
    -- Metadata
    cluster_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(255) NOT NULL
);

-- Hotspot indices
CREATE INDEX idx_hotspots_geom ON hotspots USING GIST(center_geom);
CREATE INDEX idx_hotspots_h3 ON hotspots(center_h3_index);
CREATE INDEX idx_hotspots_time ON hotspots(time_window_start, time_window_end);
CREATE INDEX idx_hotspots_significance ON hotspots(statistical_significance) WHERE statistical_significance < 0.05;

-- Predictions table for ML model outputs
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id VARCHAR(255) NOT NULL,
    
    -- Prediction details
    predicted_geom GEOMETRY(POINT, 4326) NOT NULL,
    predicted_h3_index VARCHAR(15) NOT NULL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    
    -- Model information
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL,
    prediction_horizon_hours INTEGER NOT NULL,
    
    -- Risk assessment
    risk_score DECIMAL(5, 4) DEFAULT NULL,
    risk_factors TEXT[] DEFAULT '{}',
    
    -- Metadata
    model_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(255) NOT NULL
);

-- Prediction indices
CREATE INDEX idx_predictions_entity_time ON predictions(entity_id, prediction_timestamp);
CREATE INDEX idx_predictions_geom ON predictions USING GIST(predicted_geom);
CREATE INDEX idx_predictions_h3 ON predictions(predicted_h3_index);
CREATE INDEX idx_predictions_confidence ON predictions(confidence_score DESC);

-- Anomalies table for detected unusual patterns
CREATE TABLE anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    anomaly_id VARCHAR(255) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    
    -- Anomaly details
    anomaly_type VARCHAR(50) NOT NULL,
    anomaly_geom GEOMETRY(POINT, 4326) NOT NULL,
    anomaly_h3_index VARCHAR(15) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Scoring
    anomaly_score DECIMAL(5, 4) NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    
    -- Context
    description TEXT NOT NULL,
    context_data JSONB DEFAULT '{}',
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ DEFAULT NULL,
    resolution_notes TEXT DEFAULT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    tenant_id VARCHAR(255) NOT NULL
);

-- Anomaly indices
CREATE INDEX idx_anomalies_entity_time ON anomalies(entity_id, timestamp DESC);
CREATE INDEX idx_anomalies_geom ON anomalies USING GIST(anomaly_geom);
CREATE INDEX idx_anomalies_h3 ON anomalies(anomaly_h3_index);
CREATE INDEX idx_anomalies_type ON anomalies(anomaly_type);
CREATE INDEX idx_anomalies_severity ON anomalies(severity_level);
CREATE INDEX idx_anomalies_unresolved ON anomalies(resolved, timestamp DESC) WHERE NOT resolved;

-- Performance optimization views
CREATE MATERIALIZED VIEW entity_location_summary AS
SELECT 
    entity_id,
    entity_type,
    tenant_id,
    COUNT(*) as total_points,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen,
    ST_Centroid(ST_Collect(geom)) as centroid_geom,
    array_agg(DISTINCT primary_h3_index) as h3_cells_visited,
    AVG(speed_kmh) as avg_speed_kmh,
    MAX(speed_kmh) as max_speed_kmh
FROM coordinates
GROUP BY entity_id, entity_type, tenant_id;

CREATE INDEX idx_entity_summary_entity ON entity_location_summary(entity_id);
CREATE INDEX idx_entity_summary_type ON entity_location_summary(entity_type);
CREATE INDEX idx_entity_summary_centroid ON entity_location_summary USING GIST(centroid_geom);

-- Refresh materialized view periodically
CREATE OR REPLACE FUNCTION refresh_entity_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY entity_location_summary;
END;
$$ LANGUAGE plpgsql;

-- Partitioning for large datasets
CREATE TABLE coordinates_partitioned (
    LIKE coordinates INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions (example for 2025)
CREATE TABLE coordinates_2025_01 PARTITION OF coordinates_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE coordinates_2025_02 PARTITION OF coordinates_partitioned
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
-- ... continue for other months

-- Triggers for automatic data quality and maintenance
CREATE OR REPLACE FUNCTION update_coordinate_metadata()
RETURNS TRIGGER AS $$
BEGIN
    -- Update timestamp
    NEW.updated_at = NOW();
    
    -- Generate geohash if not provided
    IF NEW.geohash IS NULL THEN
        NEW.geohash = calculate_geohash(NEW.latitude, NEW.longitude, 8);
    END IF;
    
    -- Validate H3 indices
    IF NEW.h3_indices = '{}'::jsonb THEN
        -- Generate H3 indices (would call custom function)
        NEW.h3_indices = generate_h3_indices_sql(NEW.latitude, NEW.longitude);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_coordinate_metadata
    BEFORE INSERT OR UPDATE ON coordinates
    FOR EACH ROW
    EXECUTE FUNCTION update_coordinate_metadata();

-- Data retention policy
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete coordinates older than 2 years
    DELETE FROM coordinates 
    WHERE timestamp < NOW() - INTERVAL '2 years';
    
    -- Delete resolved anomalies older than 1 year
    DELETE FROM anomalies 
    WHERE resolved = TRUE 
    AND resolved_at < NOW() - INTERVAL '1 year';
    
    -- Clean up old predictions
    DELETE FROM predictions 
    WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;
```

### Query Optimization Strategies

```python
class DatabaseOptimizer:
    """Database query optimization for geographical queries."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def optimized_spatial_query(
        self,
        center_lat: float,
        center_lng: float,
        radius_meters: float,
        entity_type: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> List[GLSCoordinate]:
        """Optimized spatial query using multiple indexing strategies."""
        
        # Choose optimal query strategy based on radius
        if radius_meters <= 100:
            # Very small radius - use H3 index at high resolution
            return await self._h3_spatial_query(center_lat, center_lng, radius_meters, 8)
        elif radius_meters <= 1000:
            # Medium radius - use H3 index at medium resolution
            return await self._h3_spatial_query(center_lat, center_lng, radius_meters, 6)
        elif radius_meters <= 10000:
            # Large radius - use PostGIS spatial index
            return await self._postgis_spatial_query(center_lat, center_lng, radius_meters, entity_type, time_window_hours)
        else:
            # Very large radius - use materialized view
            return await self._summary_spatial_query(center_lat, center_lng, radius_meters)
    
    async def _h3_spatial_query(
        self, 
        center_lat: float, 
        center_lng: float, 
        radius_meters: float, 
        h3_resolution: int
    ) -> List[GLSCoordinate]:
        """Use H3 index for spatial querying."""
        
        # Generate center H3 index
        center_h3 = generate_h3_index(center_lat, center_lng, h3_resolution)
        
        # Calculate ring size based on radius
        h3_edge_length = H3_EDGE_LENGTHS[h3_resolution]  # meters
        ring_size = max(1, int(radius_meters / h3_edge_length))
        
        # Get all H3 cells in range
        h3_cells = get_h3_ring_cells(center_h3, ring_size)
        
        # Query database using H3 index
        query = f"""
            SELECT entity_id, latitude, longitude, h3_indices, timestamp, metadata
            FROM coordinates
            WHERE (h3_indices->>'{h3_resolution}') = ANY($1)
            AND ST_DWithin(
                geom, 
                ST_Point($2, $3)::geography, 
                $4
            )
            ORDER BY ST_Distance(geom, ST_Point($2, $3)) ASC
            LIMIT 1000
        """
        
        results = await self.db.fetch_all(
            query, 
            h3_cells, 
            center_lng, 
            center_lat, 
            radius_meters
        )
        
        return [GLSCoordinate.from_db_row(row) for row in results]
    
    async def _postgis_spatial_query(
        self,
        center_lat: float,
        center_lng: float, 
        radius_meters: float,
        entity_type: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> List[GLSCoordinate]:
        """Use PostGIS spatial index for querying."""
        
        where_conditions = ["ST_DWithin(geom, ST_Point($2, $3)::geography, $4)"]
        params = [center_lng, center_lat, radius_meters]
        param_count = 4
        
        if entity_type:
            where_conditions.append(f"entity_type = ${param_count + 1}")
            params.append(entity_type)
            param_count += 1
        
        if time_window_hours:
            where_conditions.append(f"timestamp >= NOW() - INTERVAL '{time_window_hours} hours'")
        
        query = f"""
            SELECT entity_id, latitude, longitude, h3_indices, timestamp, metadata
            FROM coordinates
            WHERE {' AND '.join(where_conditions)}
            ORDER BY ST_Distance(geom, ST_Point($2, $3)) ASC
            LIMIT 1000
        """
        
        results = await self.db.fetch_all(query, *params)
        return [GLSCoordinate.from_db_row(row) for row in results]
    
    async def batch_insert_coordinates(
        self, 
        coordinates: List[GLSCoordinate]
    ) -> None:
        """Optimized batch insertion of coordinates."""
        
        if not coordinates:
            return
        
        # Prepare batch data
        batch_data = []
        for coord in coordinates:
            batch_data.append((
                coord.entity_id,
                coord.entity_type,
                coord.latitude,
                coord.longitude,
                coord.h3_indices,
                coord.timestamp,
                coord.metadata
            ))
        
        # Use COPY for maximum performance
        query = """
            INSERT INTO coordinates 
            (entity_id, entity_type, latitude, longitude, h3_indices, timestamp, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        await self.db.executemany(query, batch_data)
    
    async def create_spatial_index_if_needed(self, table_name: str, column_name: str):
        """Create spatial index if it doesn't exist."""
        
        index_name = f"idx_{table_name}_{column_name}_gist"
        
        check_query = """
            SELECT 1 FROM pg_indexes 
            WHERE tablename = $1 AND indexname = $2
        """
        
        exists = await self.db.fetch_val(check_query, table_name, index_name)
        
        if not exists:
            create_query = f"""
                CREATE INDEX CONCURRENTLY {index_name} 
                ON {table_name} USING GIST({column_name})
            """
            await self.db.execute(create_query)
```

---

*© 2025 Datacraft. All rights reserved. This technical guide is part of the APG Platform documentation.*