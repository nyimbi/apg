"""
GDELT Geographic Data Enhancement Using Central Utilities
========================================================

Refactored GDELT geographic data enhancement that leverages the central
geocoding utilities instead of re-implementing capabilities.

This module provides GDELT-specific geographic processing while using
the comprehensive geocoding utilities from packages_enhanced.utils.geocoding
for core functionality like coordinate validation, distance calculations,
and geocoding operations.

Key Features:
- **Central Utility Integration**: Uses packages_enhanced.utils.geocoding
- **GDELT-Specific Processing**: Tailored for GDELT event data structures
- **Enhanced Caching**: Leverages central caching utilities
- **Comprehensive Validation**: Uses central validation framework
- **Geographic Clustering**: Spatial clustering for hotspot analysis

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 2.0.0 (Refactored)
License: MIT
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import json

# Import central utilities instead of re-implementing
from ....utils.geocoding import (
    EnhancedGeocoder,
    GeocodingResult,
    create_geocoder,
    validate_coordinates,
    calculate_distance,
    comprehensive_geocode
)
from ....utils.caching import CacheManager
from ....utils.validation import CoordinateValidator
from ....utils.spatial import SpatialAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GDELTLocationInfo:
    """GDELT-specific location information."""
    latitude: float
    longitude: float
    country: Optional[str] = None
    country_code: Optional[str] = None
    admin1: Optional[str] = None  # State/Province
    admin2: Optional[str] = None  # County/District
    city: Optional[str] = None
    place_name: Optional[str] = None
    confidence: float = 0.0
    source: Optional[str] = None

    @classmethod
    def from_geocoding_result(cls, result: GeocodingResult) -> 'GDELTLocationInfo':
        """Create GDELT location info from central geocoding result."""
        return cls(
            latitude=result.latitude,
            longitude=result.longitude,
            country=result.country,
            country_code=result.country_code,
            admin1=result.admin_level_1,
            admin2=result.admin_level_2,
            city=result.city,
            place_name=result.formatted_address,
            confidence=result.confidence,
            source=result.provider
        )


@dataclass
class GeographicCluster:
    """Geographic cluster for hotspot analysis."""
    center_lat: float
    center_lon: float
    radius_km: float
    event_count: int
    events: List[Dict[str, Any]]


class GDELTLocationEnhancer:
    """
    GDELT location enhancer using central geocoding utilities.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize with optional cache manager.

        Args:
            cache_manager: Optional cache manager instance
        """
        self.geocoder = None
        self.cache_manager = cache_manager
        self._initialized = False

    async def initialize(self):
        """Initialize the geocoder and cache manager."""
        if self._initialized:
            return

        # Create geocoder with multiple providers and caching
        self.geocoder = create_geocoder(
            providers=['nominatim', 'google', 'mapbox'],
            cache_backend='memory' if not self.cache_manager else 'external',
            enable_validation=True,
            timeout=30
        )

        # Initialize cache manager if not provided
        if not self.cache_manager:
            self.cache_manager = await CacheManager.create(
                strategy='lru',
                max_size=10000,
                ttl=3600  # 1 hour TTL for geocoding results
            )

        self._initialized = True
        logger.info("GDELT Location Enhancer initialized with central utilities")

    async def enhance_location(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None,
        country_code: Optional[str] = None
    ) -> Optional[GDELTLocationInfo]:
        """
        Enhance location data using central geocoding utilities.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            location_name: Location name for geocoding
            country_code: Country code for context

        Returns:
            Enhanced location information or None
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Case 1: We have coordinates, do reverse geocoding
            if latitude is not None and longitude is not None:
                if validate_coordinates(latitude, longitude):
                    result = await self.geocoder.reverse_geocode(
                        latitude=latitude,
                        longitude=longitude
                    )
                    if result:
                        return GDELTLocationInfo.from_geocoding_result(result)
                else:
                    logger.warning(f"Invalid coordinates: {latitude}, {longitude}")
                    return None

            # Case 2: We have location name, do forward geocoding
            elif location_name:
                # Use comprehensive geocoding for better results
                result = await comprehensive_geocode(
                    address=location_name,
                    country_hint=country_code,
                    include_admin_boundaries=True,
                    cache_manager=self.cache_manager
                )
                if result:
                    return GDELTLocationInfo.from_geocoding_result(result)

            return None

        except Exception as e:
            logger.error(f"Location enhancement failed: {e}")
            return None

    async def batch_enhance_locations(
        self,
        locations: List[Dict[str, Any]]
    ) -> List[Optional[GDELTLocationInfo]]:
        """
        Batch enhance multiple locations efficiently.

        Args:
            locations: List of location dictionaries

        Returns:
            List of enhanced location info objects
        """
        if not self._initialized:
            await self.initialize()

        # Prepare batch geocoding requests
        geocoding_requests = []
        coordinate_requests = []

        for i, loc in enumerate(locations):
            lat = loc.get('latitude')
            lon = loc.get('longitude')
            name = loc.get('location_name')

            if lat is not None and lon is not None and validate_coordinates(lat, lon):
                coordinate_requests.append((i, lat, lon))
            elif name:
                geocoding_requests.append((i, name, loc.get('country_code')))

        results = [None] * len(locations)

        # Process coordinate-based requests (reverse geocoding)
        if coordinate_requests:
            reverse_tasks = [
                self.geocoder.reverse_geocode(latitude=lat, longitude=lon)
                for _, lat, lon in coordinate_requests
            ]
            reverse_results = await asyncio.gather(*reverse_tasks, return_exceptions=True)

            for (i, _, _), result in zip(coordinate_requests, reverse_results):
                if isinstance(result, GeocodingResult):
                    results[i] = GDELTLocationInfo.from_geocoding_result(result)

        # Process name-based requests (forward geocoding)
        if geocoding_requests:
            addresses = [name for _, name, _ in geocoding_requests]
            geocoding_results = await self.geocoder.batch_geocode(addresses)

            for (i, _, _), result in zip(geocoding_requests, geocoding_results):
                if result:
                    results[i] = GDELTLocationInfo.from_geocoding_result(result)

        return results


class GDELTGeocoder:
    """Main geocoding interface for GDELT data using central utilities."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize GDELT geocoder.

        Args:
            cache_manager: Optional cache manager instance
        """
        self.enhancer = GDELTLocationEnhancer(cache_manager)
        self.spatial_analyzer = SpatialAnalyzer()
        self._initialized = False

    async def initialize(self):
        """Initialize the geocoder components."""
        if not self._initialized:
            await self.enhancer.initialize()
            self._initialized = True

    async def process_event_location(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and enhance location data for a GDELT event.

        Args:
            event_data: GDELT event dictionary

        Returns:
            Enhanced event data with location information
        """
        if not self._initialized:
            await self.initialize()

        # Extract location data
        latitude = event_data.get('latitude')
        longitude = event_data.get('longitude')
        location_name = event_data.get('location_name')
        country = event_data.get('country')

        # Enhance location using central utilities
        enhanced_location = await self.enhancer.enhance_location(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            country_code=country
        )

        # Update event data
        if enhanced_location:
            event_data.update({
                'enhanced_location': {
                    'latitude': enhanced_location.latitude,
                    'longitude': enhanced_location.longitude,
                    'country': enhanced_location.country,
                    'country_code': enhanced_location.country_code,
                    'admin1': enhanced_location.admin1,
                    'admin2': enhanced_location.admin2,
                    'city': enhanced_location.city,
                    'place_name': enhanced_location.place_name,
                    'confidence': enhanced_location.confidence,
                    'source': enhanced_location.source
                }
            })

        return event_data

    async def cluster_events_geographically(
        self,
        events: List[Dict[str, Any]],
        radius_km: float = 50.0,
        min_events: int = 3
    ) -> List[GeographicCluster]:
        """
        Cluster events geographically for hotspot analysis using central spatial utilities.

        Args:
            events: List of GDELT events
            radius_km: Clustering radius in kilometers
            min_events: Minimum events per cluster

        Returns:
            List of geographic clusters
        """
        if not self._initialized:
            await self.initialize()

        # Extract coordinates from events
        coordinates = []
        valid_events = []

        for event in events:
            enhanced_loc = event.get('enhanced_location', {})
            lat = enhanced_loc.get('latitude') or event.get('latitude')
            lon = enhanced_loc.get('longitude') or event.get('longitude')

            if lat is not None and lon is not None and validate_coordinates(lat, lon):
                coordinates.append((lat, lon))
                valid_events.append(event)

        if len(coordinates) < min_events:
            return []

        # Use spatial analyzer for clustering
        clusters = await self.spatial_analyzer.cluster_points_spatial(
            coordinates=coordinates,
            radius_km=radius_km,
            min_points=min_events
        )

        # Convert to GDELT geographic clusters
        gdelt_clusters = []
        for cluster in clusters:
            if len(cluster.point_indices) >= min_events:
                cluster_events = [valid_events[i] for i in cluster.point_indices]

                gdelt_cluster = GeographicCluster(
                    center_lat=cluster.centroid[0],
                    center_lon=cluster.centroid[1],
                    radius_km=cluster.radius_km,
                    event_count=len(cluster_events),
                    events=cluster_events
                )
                gdelt_clusters.append(gdelt_cluster)

        return gdelt_clusters

    async def batch_process_events(
        self,
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch process multiple events efficiently.

        Args:
            events: List of GDELT events

        Returns:
            List of enhanced events
        """
        if not self._initialized:
            await self.initialize()

        # Extract location data for batch processing
        locations = []
        for event in events:
            locations.append({
                'latitude': event.get('latitude'),
                'longitude': event.get('longitude'),
                'location_name': event.get('location_name'),
                'country_code': event.get('country')
            })

        # Batch enhance locations
        enhanced_locations = await self.enhancer.batch_enhance_locations(locations)

        # Update events with enhanced locations
        enhanced_events = []
        for event, enhanced_loc in zip(events, enhanced_locations):
            if enhanced_loc:
                event.update({
                    'enhanced_location': {
                        'latitude': enhanced_loc.latitude,
                        'longitude': enhanced_loc.longitude,
                        'country': enhanced_loc.country,
                        'country_code': enhanced_loc.country_code,
                        'admin1': enhanced_loc.admin1,
                        'admin2': enhanced_loc.admin2,
                        'city': enhanced_loc.city,
                        'place_name': enhanced_loc.place_name,
                        'confidence': enhanced_loc.confidence,
                        'source': enhanced_loc.source
                    }
                })
            enhanced_events.append(event)

        return enhanced_events


# Factory functions for convenience
async def create_gdelt_geocoder(cache_config: Optional[Dict[str, Any]] = None) -> GDELTGeocoder:
    """
    Create and initialize a GDELT geocoder.

    Args:
        cache_config: Optional cache configuration

    Returns:
        Initialized GDELT geocoder
    """
    cache_manager = None
    if cache_config:
        cache_manager = await CacheManager.create(**cache_config)

    geocoder = GDELTGeocoder(cache_manager)
    await geocoder.initialize()
    return geocoder


# Convenience functions using central utilities
def validate_gdelt_coordinates(latitude: float, longitude: float) -> bool:
    """Validate GDELT coordinates using central validation."""
    return validate_coordinates(latitude, longitude)


def calculate_gdelt_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    unit: str = "km"
) -> float:
    """Calculate distance between GDELT event locations."""
    return calculate_distance(lat1, lon1, lat2, lon2, unit)


# Export the main components
__all__ = [
    'GDELTLocationInfo',
    'GeographicCluster',
    'GDELTLocationEnhancer',
    'GDELTGeocoder',
    'create_gdelt_geocoder',
    'validate_gdelt_coordinates',
    'calculate_gdelt_distance'
]
