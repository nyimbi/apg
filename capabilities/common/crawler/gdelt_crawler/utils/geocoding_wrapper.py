"""
GDELT Geocoding Wrapper
=======================

A wrapper around the utils/geocoding package that provides GDELT-specific
geocoding functionality while leveraging the existing geocoding infrastructure.

This module replaces the custom geocoding implementation with proper integration
of the packages_enhanced/utils/geocoding system.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import from utils packages
from ....utils.geocoding import GeocodingManager, LocationInfo as BaseLocationInfo
from ....utils.spatial import SpatialOperations, GeographicCluster as BaseSpatialCluster
from ....utils.spatial.operations import haversine_distance

logger = logging.getLogger(__name__)


class GDELTLocationInfo(BaseLocationInfo):
    """GDELT-specific extension of LocationInfo."""
    
    def __init__(self, **kwargs):
        # Add GDELT-specific fields
        self.goldstein_scale = kwargs.pop('goldstein_scale', None)
        self.avg_tone = kwargs.pop('avg_tone', None)
        self.event_code = kwargs.pop('event_code', None)
        super().__init__(**kwargs)
    
    def to_gdelt_dict(self) -> Dict[str, Any]:
        """Convert to GDELT-compatible dictionary."""
        base_dict = self.to_dict()
        base_dict.update({
            'goldstein_scale': self.goldstein_scale,
            'avg_tone': self.avg_tone,
            'event_code': self.event_code
        })
        return base_dict


class GDELTGeographicCluster(BaseSpatialCluster):
    """GDELT-specific geographic cluster."""
    
    def __init__(self, **kwargs):
        self.event_codes = kwargs.pop('event_codes', [])
        self.avg_goldstein = kwargs.pop('avg_goldstein', 0.0)
        self.dominant_location = kwargs.pop('dominant_location', None)
        super().__init__(**kwargs)


class GDELTGeocodingWrapper:
    """
    Wrapper for GDELT-specific geocoding operations using utils/geocoding.
    
    This replaces the custom geocoding implementation with proper integration
    of the existing geocoding infrastructure.
    """
    
    def __init__(self, geocoding_config: Optional[Dict[str, Any]] = None):
        """
        Initialize GDELT geocoding wrapper.
        
        Args:
            geocoding_config: Configuration for geocoding manager
        """
        # Initialize geocoding manager from utils
        config = geocoding_config or {
            'default_provider': 'nominatim',
            'enable_caching': True,
            'cache_ttl': 86400,  # 24 hours
            'rate_limit': 1.0  # 1 request per second for Nominatim
        }
        
        self.geocoding_manager = GeocodingManager(config)
        self.spatial_ops = SpatialOperations()
        
        # GDELT-specific country mappings (extend base mappings)
        self.gdelt_country_codes = {
            'US': 'USA',  # GDELT uses USA instead of US
            'UK': 'GBR',  # GDELT uses UK for United Kingdom
            # Add more GDELT-specific mappings as needed
        }
    
    async def enhance_location(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        location_name: Optional[str] = None,
        country_code: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None
    ) -> GDELTLocationInfo:
        """
        Enhance location information for GDELT events.
        
        Args:
            latitude: Event latitude
            longitude: Event longitude
            location_name: Location name from GDELT
            country_code: Country code from GDELT
            event_data: Additional GDELT event data
            
        Returns:
            Enhanced GDELT location information
        """
        # Use geocoding manager for validation and enhancement
        if latitude is not None and longitude is not None:
            # Validate coordinates
            if self.geocoding_manager.validate_coordinates(latitude, longitude):
                # Get enhanced location info
                base_location = await self.geocoding_manager.reverse_geocode(
                    latitude, longitude
                )
                
                if base_location:
                    # Create GDELT-specific location info
                    gdelt_location = GDELTLocationInfo(
                        latitude=base_location.latitude,
                        longitude=base_location.longitude,
                        address=base_location.address,
                        city=base_location.city,
                        state_province=base_location.state_province,
                        country=base_location.country,
                        country_code=self._normalize_country_code(
                            base_location.country_code or country_code
                        ),
                        postal_code=base_location.postal_code,
                        confidence_score=base_location.confidence_score,
                        provider=base_location.provider,
                        raw_response=base_location.raw_response,
                        metadata=base_location.metadata
                    )
                    
                    # Add GDELT-specific data
                    if event_data:
                        gdelt_location.goldstein_scale = event_data.get('goldstein_scale')
                        gdelt_location.avg_tone = event_data.get('avg_tone')
                        gdelt_location.event_code = event_data.get('event_code')
                    
                    return gdelt_location
        
        # Fallback: create location from available data
        return GDELTLocationInfo(
            latitude=latitude or 0.0,
            longitude=longitude or 0.0,
            address=location_name,
            country_code=self._normalize_country_code(country_code),
            confidence_score=0.5 if location_name else 0.1,
            provider='gdelt_raw'
        )
    
    async def geocode_location_name(self, location_name: str) -> Optional[GDELTLocationInfo]:
        """
        Geocode a location name to coordinates.
        
        Args:
            location_name: Location name to geocode
            
        Returns:
            GDELT location info or None
        """
        result = await self.geocoding_manager.geocode(location_name)
        
        if result:
            return GDELTLocationInfo(
                latitude=result.latitude,
                longitude=result.longitude,
                address=result.address,
                city=result.city,
                state_province=result.state_province,
                country=result.country,
                country_code=self._normalize_country_code(result.country_code),
                postal_code=result.postal_code,
                confidence_score=result.confidence_score,
                provider=result.provider,
                raw_response=result.raw_response,
                metadata=result.metadata
            )
        
        return None
    
    def cluster_events_geographically(
        self,
        events: List[Dict[str, Any]],
        radius_km: float = 50.0,
        min_events: int = 3
    ) -> List[GDELTGeographicCluster]:
        """
        Cluster GDELT events geographically.
        
        Args:
            events: List of GDELT events with coordinates
            radius_km: Clustering radius in kilometers
            min_events: Minimum events to form a cluster
            
        Returns:
            List of GDELT geographic clusters
        """
        # Convert events to spatial points
        points = []
        event_map = {}
        
        for i, event in enumerate(events):
            lat = event.get('latitude')
            lon = event.get('longitude')
            
            if lat is not None and lon is not None:
                if self.geocoding_manager.validate_coordinates(lat, lon):
                    point_id = f"event_{i}"
                    points.append({
                        'id': point_id,
                        'latitude': lat,
                        'longitude': lon
                    })
                    event_map[point_id] = event
        
        if len(points) < min_events:
            return []
        
        # Use spatial operations for clustering
        clusters = self.spatial_ops.cluster_points(
            points=points,
            radius_km=radius_km,
            min_points=min_events
        )
        
        # Convert to GDELT clusters
        gdelt_clusters = []
        
        for cluster in clusters:
            cluster_events = [event_map[pid] for pid in cluster.point_ids if pid in event_map]
            
            # Calculate GDELT-specific cluster properties
            event_codes = [e.get('event_code') for e in cluster_events if e.get('event_code')]
            goldstein_values = [e.get('goldstein_scale') for e in cluster_events 
                              if e.get('goldstein_scale') is not None]
            
            avg_goldstein = sum(goldstein_values) / len(goldstein_values) if goldstein_values else 0.0
            
            # Find dominant location
            location_names = [e.get('location_name') for e in cluster_events 
                            if e.get('location_name')]
            dominant_location = max(set(location_names), key=location_names.count) if location_names else None
            
            gdelt_cluster = GDELTGeographicCluster(
                cluster_id=cluster.cluster_id,
                center_latitude=cluster.center_latitude,
                center_longitude=cluster.center_longitude,
                radius_km=radius_km,
                point_ids=cluster.point_ids,
                point_count=cluster.point_count,
                created_at=cluster.created_at,
                metadata=cluster.metadata,
                event_codes=list(set(event_codes)),
                avg_goldstein=avg_goldstein,
                dominant_location=dominant_location
            )
            
            gdelt_clusters.append(gdelt_cluster)
        
        return sorted(gdelt_clusters, key=lambda c: c.point_count, reverse=True)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        return haversine_distance(lat1, lon1, lat2, lon2)
    
    def find_nearby_events(
        self,
        center_lat: float,
        center_lon: float,
        events: List[Dict[str, Any]],
        radius_km: float
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find events within a radius of a center point.
        
        Args:
            center_lat, center_lon: Center point coordinates
            events: List of events to search
            radius_km: Search radius in kilometers
            
        Returns:
            List of (event, distance) tuples within radius
        """
        nearby = []
        
        for event in events:
            lat = event.get('latitude')
            lon = event.get('longitude')
            
            if lat is not None and lon is not None:
                distance = self.calculate_distance(center_lat, center_lon, lat, lon)
                if distance <= radius_km:
                    nearby.append((event, distance))
        
        return sorted(nearby, key=lambda x: x[1])  # Sort by distance
    
    def _normalize_country_code(self, country_code: Optional[str]) -> Optional[str]:
        """Normalize GDELT country codes to ISO standards."""
        if not country_code:
            return None
        
        # Check GDELT-specific mappings first
        if country_code in self.gdelt_country_codes:
            return self.gdelt_country_codes[country_code]
        
        # Return as-is if not in mappings
        return country_code.upper()
    
    async def batch_enhance_locations(
        self,
        events: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Enhance locations for multiple events in batches.
        
        Args:
            events: List of GDELT events
            batch_size: Batch size for processing
            
        Returns:
            Events with enhanced location data
        """
        enhanced_events = []
        
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            
            for event in batch:
                try:
                    enhanced_location = await self.enhance_location(
                        latitude=event.get('latitude'),
                        longitude=event.get('longitude'),
                        location_name=event.get('location_name'),
                        country_code=event.get('country'),
                        event_data=event
                    )
                    
                    # Update event with enhanced location
                    event['enhanced_location'] = enhanced_location.to_gdelt_dict()
                    enhanced_events.append(event)
                    
                except Exception as e:
                    logger.error(f"Failed to enhance location for event: {e}")
                    enhanced_events.append(event)
        
        return enhanced_events


# Backward compatibility - create instance
_default_wrapper = None

def get_gdelt_geocoder() -> GDELTGeocodingWrapper:
    """Get default GDELT geocoder instance."""
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = GDELTGeocodingWrapper()
    return _default_wrapper


# Utility functions for backward compatibility
async def enhance_event_location(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance location data for a GDELT event."""
    geocoder = get_gdelt_geocoder()
    enhanced = await geocoder.enhance_location(
        latitude=event_data.get('latitude'),
        longitude=event_data.get('longitude'),
        location_name=event_data.get('location_name'),
        country_code=event_data.get('country'),
        event_data=event_data
    )
    event_data['enhanced_location'] = enhanced.to_gdelt_dict()
    return event_data


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers."""
    return haversine_distance(lat1, lon1, lat2, lon2)


# Export components
__all__ = [
    'GDELTGeocodingWrapper',
    'GDELTLocationInfo',
    'GDELTGeographicCluster',
    'get_gdelt_geocoder',
    'enhance_event_location',
    'calculate_distance'
]