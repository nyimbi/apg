# GDELT Crawler Migration Guide

## Migrating from Local Implementations to Central Utilities

This guide outlines how to migrate the GDELT crawler from using local utility implementations to the comprehensive utilities available in `packages_enhanced/utils`. This migration will eliminate code duplication, improve maintainability, and provide access to more advanced features.

## Overview of Changes

The GDELT crawler currently re-implements several capabilities that are already available in the central utilities:

1. **Geocoding** - Local implementation in `utils/geocoding.py`
2. **Caching** - Basic dictionary-based caching in API client
3. **Monitoring** - Custom metrics collection
4. **Validation** - Coordinate validation logic

## Migration Steps

### Step 1: Update Geocoding Implementation

#### Current Implementation Issues
- **File**: `gdelt_crawler/utils/geocoding.py`
- **Issues**: 
  - Re-implements coordinate validation
  - Basic Haversine distance calculation
  - Simple reverse geocoding with limited providers
  - Basic dictionary caching

#### Migration Actions

1. **Replace coordinate validation**:
   ```python
   # OLD (local implementation)
   from .utils.geocoding import CoordinateValidator
   validator = CoordinateValidator()
   is_valid = validator.is_valid_coordinate(lat, lon)
   
   # NEW (central utility)
   from ....utils.geocoding import validate_coordinates
   is_valid = validate_coordinates(lat, lon)
   ```

2. **Replace distance calculations**:
   ```python
   # OLD (local implementation)
   from .utils.geocoding import DistanceCalculator
   calc = DistanceCalculator()
   distance = calc.haversine_distance(lat1, lon1, lat2, lon2)
   
   # NEW (central utility)
   from ....utils.geocoding import calculate_distance
   distance = calculate_distance(lat1, lon1, lat2, lon2, unit="km")
   ```

3. **Replace geocoding operations**:
   ```python
   # OLD (local implementation)
   from .utils.geocoding import LocationEnhancer
   enhancer = LocationEnhancer()
   location = enhancer.enhance_location(lat, lon, name, country)
   
   # NEW (central utility)
   from ....utils.geocoding import create_geocoder, comprehensive_geocode
   geocoder = create_geocoder(providers=['nominatim', 'google', 'mapbox'])
   result = await geocoder.reverse_geocode(latitude=lat, longitude=lon)
   # or for comprehensive geocoding
   result = await comprehensive_geocode(address=name, country_hint=country)
   ```

#### Benefits of Migration
- **Multi-provider support**: Automatic fallback between geocoding providers
- **Advanced caching**: Hierarchical caching with TTL and LRU policies
- **Batch processing**: Efficient batch geocoding operations
- **Quality assessment**: Confidence scoring and result validation
- **H3 integration**: Spatial indexing for improved performance

### Step 2: Update Caching Implementation

#### Current Implementation Issues
- **Files**: `api/gdelt_client.py`
- **Issues**:
  - Simple dictionary-based caching (`self._cache`)
  - Basic TTL implementation
  - No cache analytics or optimization
  - Memory-only storage

#### Migration Actions

1. **Replace API client caching**:
   ```python
   # OLD (local implementation)
   self._cache: Dict[str, Tuple[List[GDELTArticle], float]] = {}
   self._cache_ttl = 3600
   
   # Check cache
   if cache_key in self._cache:
       cached_data, timestamp = self._cache[cache_key]
       if self._is_cache_valid(timestamp):
           return cached_data
   
   # NEW (central utility)
   from ....utils.caching import CacheManager, cache_decorator
   
   # Initialize cache manager
   self.cache_manager = await CacheManager.create(
       strategy='lru',
       max_size=10000,
       ttl=3600,
       backend='memory'  # or 'redis' for persistent caching
   )
   
   # Use decorator for automatic caching
   @cache_decorator(ttl=3600, key_prefix='gdelt_query')
   async def fetch_articles(self, query_params):
       # API call logic here
       pass
   ```

2. **Enable advanced caching features**:
   ```python
   # Spatial caching for geographic queries
   from ....utils.caching import spatial_cache_decorator
   
   @spatial_cache_decorator(
       ttl=7200,
       spatial_key='coordinates',
       radius_km=50
   )
   async def query_by_location(self, lat, lon, radius):
       # Location-based query logic
       pass
   ```

#### Benefits of Migration
- **Multiple backends**: Memory, Redis, file system, database
- **Advanced strategies**: LRU, LFU, TTL, spatial-aware, adaptive
- **Performance monitoring**: Cache hit rates, optimization suggestions
- **Compression & encryption**: Data security and storage efficiency
- **Distributed caching**: Multi-node cache synchronization

### Step 3: Update Monitoring Implementation

#### Current Implementation Issues
- **Files**: `monitoring/metrics.py`, `monitoring/alerts.py`
- **Issues**:
  - Custom MetricPoint class
  - Basic performance tracking
  - Limited system metrics collection

#### Migration Actions

1. **Replace metrics collection**:
   ```python
   # OLD (local implementation)
   from .monitoring.metrics import GDELTMetrics, MetricPoint
   metrics = GDELTMetrics()
   metrics.record_metric('api_calls', 1)
   
   # NEW (central utility)
   from ....utils.monitoring import get_monitoring_manager, monitor_performance
   
   # Initialize monitoring
   monitoring = get_monitoring_manager()
   await monitoring.initialize()
   
   # Use decorator for automatic performance monitoring
   @monitor_performance('gdelt_api_call')
   async def api_call_method(self):
       # API call logic
       pass
   ```

2. **Enable advanced monitoring features**:
   ```python
   # Context-based monitoring
   from ....utils.monitoring import monitoring_context
   
   async with monitoring_context('gdelt_batch_processing') as ctx:
       # Batch processing logic
       ctx.add_metric('events_processed', len(events))
       ctx.add_metric('processing_time', time.time() - start_time)
   ```

#### Benefits of Migration
- **Prometheus integration**: Enterprise-grade metrics collection
- **Advanced profiling**: Detailed performance analysis
- **Health monitoring**: Component health checks and status
- **Memory leak detection**: Automatic resource monitoring
- **Real-time dashboards**: Live monitoring capabilities

### Step 4: File Structure Changes

#### Files to Modify

1. **`__init__.py`** - Update imports to use central utilities
2. **`api/gdelt_client.py`** - Replace caching implementation
3. **`database/etl.py`** - Update to use central monitoring
4. **`bulk/file_downloader.py`** - Integrate central caching and monitoring

#### Files to Remove (after migration)

1. **`utils/geocoding.py`** - Replace with central geocoding utilities
2. **`monitoring/metrics.py`** - Replace with central monitoring
3. **Consider consolidating `monitoring/alerts.py`** with central notification utilities

### Step 5: Update Import Statements

#### Global Import Updates

```python
# Update all files to use central utilities instead of local implementations

# OLD imports
from .utils.geocoding import GDELTGeocoder, LocationEnhancer, coordinate_validator
from .monitoring.metrics import GDELTMetrics, MetricPoint
from .monitoring.alerts import GDELTAlertSystem

# NEW imports
from ....utils.geocoding import (
    create_geocoder, 
    validate_coordinates, 
    calculate_distance,
    comprehensive_geocode
)
from ....utils.caching import CacheManager, cache_decorator, spatial_cache_decorator
from ....utils.monitoring import get_monitoring_manager, monitor_performance
from ....utils.notification import NotificationManager
```

### Step 6: Configuration Updates

#### Update Configuration Classes

```python
# Update GDELTCrawlerConfig to include central utility configurations

@dataclass
class GDELTCrawlerConfig:
    # Existing fields...
    
    # NEW: Central utility configurations
    geocoding_config: Dict[str, Any] = field(default_factory=lambda: {
        'providers': ['nominatim', 'google', 'mapbox'],
        'timeout': 30,
        'cache_ttl': 3600,
        'enable_batch_processing': True
    })
    
    caching_config: Dict[str, Any] = field(default_factory=lambda: {
        'strategy': 'lru',
        'max_size': 50000,
        'ttl': 3600,
        'backend': 'memory',
        'compression': True,
        'enable_monitoring': True
    })
    
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_prometheus': True,
        'enable_profiling': True,
        'enable_health_checks': True,
        'metrics_interval': 60
    })
```

## Testing the Migration

### Step 1: Unit Tests Update

```python
# Update unit tests to work with central utilities
import pytest
from unittest.mock import AsyncMock, MagicMock

# Test geocoding integration
@pytest.mark.asyncio
async def test_geocoding_integration():
    from ....utils.geocoding import create_geocoder
    
    # Mock the central geocoder
    mock_geocoder = AsyncMock()
    mock_geocoder.reverse_geocode.return_value = MagicMock(
        latitude=40.7128,
        longitude=-74.0060,
        formatted_address="New York, NY, USA"
    )
    
    # Test GDELT integration
    gdelt_geocoder = GDELTGeocoder(mock_geocoder)
    result = await gdelt_geocoder.process_event_location({
        'latitude': 40.7128,
        'longitude': -74.0060
    })
    
    assert result['enhanced_location']['city'] == 'New York'
```

### Step 2: Integration Tests

```python
@pytest.mark.asyncio
async def test_caching_integration():
    from ....utils.caching import CacheManager
    
    # Test cache integration
    cache_manager = await CacheManager.create(
        strategy='lru',
        max_size=1000,
        ttl=300
    )
    
    client = EnhancedGDELTClient(cache_manager=cache_manager)
    await client.initialize()
    
    # Test cached query
    result1 = await client.query_with_caching('test query')
    result2 = await client.query_with_caching('test query')  # Should hit cache
    
    stats = await client.get_cache_statistics()
    assert stats['hit_rate'] > 0
```

## Performance Considerations

### Expected Performance Improvements

1. **Geocoding Performance**:
   - **Multi-provider fallback**: Reduced failures due to provider issues
   - **Batch processing**: Up to 5x faster for bulk geocoding operations
   - **Advanced caching**: 80-90% cache hit rates for repeated locations

2. **Caching Performance**:
   - **Memory efficiency**: Compression reduces memory usage by 60-70%
   - **Cache strategies**: Adaptive algorithms improve hit rates by 20-30%
   - **Distributed caching**: Horizontal scaling capabilities

3. **Monitoring Performance**:
   - **Prometheus integration**: Industry-standard metrics collection
   - **Reduced overhead**: Optimized monitoring with minimal performance impact

### Monitoring Migration Success

```python
# Add migration success metrics
async def track_migration_metrics():
    monitoring = get_monitoring_manager()
    
    # Track cache performance improvement
    await monitoring.add_metric('cache_hit_rate', cache_hit_rate)
    await monitoring.add_metric('geocoding_success_rate', geocoding_success_rate)
    await monitoring.add_metric('api_response_time', avg_response_time)
    
    # Track error rates
    await monitoring.add_metric('geocoding_errors', geocoding_error_count)
    await monitoring.add_metric('cache_errors', cache_error_count)
```

## Rollback Plan

### In Case of Issues

1. **Keep backup of original files** before migration
2. **Gradual migration**: Migrate one component at a time
3. **Feature flags**: Use configuration to toggle between old and new implementations
4. **Monitoring**: Set up alerts for performance degradation

```python
# Example feature flag approach
class GDELTCrawlerConfig:
    # Migration feature flags
    use_central_geocoding: bool = True
    use_central_caching: bool = True
    use_central_monitoring: bool = True
    
    def __post_init__(self):
        if not self.use_central_geocoding:
            logger.warning("Using deprecated local geocoding implementation")
```

## Timeline and Phases

### Phase 1: Preparation (Week 1)
- Set up central utilities dependencies
- Create backup of current implementation
- Update configuration classes
- Create migration test suite

### Phase 2: Geocoding Migration (Week 2)
- Replace coordinate validation logic
- Update distance calculation calls
- Migrate reverse geocoding functionality
- Test geocoding integration

### Phase 3: Caching Migration (Week 3)
- Replace API client caching
- Implement cache decorators
- Configure cache backends
- Test caching performance

### Phase 4: Monitoring Migration (Week 4)
- Replace metrics collection
- Integrate performance monitoring
- Set up health checks
- Configure alerting

### Phase 5: Testing and Optimization (Week 5)
- Run comprehensive test suite
- Performance benchmarking
- Optimization tuning
- Documentation updates

## Post-Migration Benefits

### Immediate Benefits
- **Reduced code duplication**: ~500 lines of duplicate code eliminated
- **Improved maintainability**: Single source of truth for utilities
- **Enhanced features**: Access to advanced utility capabilities
- **Better testing**: Centralized utility testing

### Long-term Benefits
- **Easier updates**: Utility improvements benefit all crawlers
- **Consistent interfaces**: Standardized APIs across all components
- **Better performance**: Optimized implementations
- **Enhanced monitoring**: Comprehensive observability

## Support and Resources

### Documentation
- [Central Geocoding Utilities](../../../utils/geocoding/README.md)
- [Caching System Documentation](../../../utils/caching/README.md)
- [Monitoring Framework Guide](../../../utils/monitoring/README.md)

### Contact
- **Author**: Nyimbi Odero
- **Company**: Datacraft (www.datacraft.co.ke)
- **Support**: Contact the development team for migration assistance

---

*This migration guide is part of the ongoing effort to improve code quality and reduce duplication across the Lindela platform.*