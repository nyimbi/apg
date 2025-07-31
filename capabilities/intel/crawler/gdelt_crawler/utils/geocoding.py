"""
GDELT Geographic Data Enhancement and Location Processing
========================================================

Geographic data enhancement utilities for GDELT events with coordinate
validation, reverse geocoding, and location standardization capabilities.

Key Features:
- **Coordinate Validation**: Robust validation of latitude/longitude pairs
- **Location Enhancement**: Reverse geocoding and location standardization
- **Geographic Clustering**: Spatial clustering for hotspot analysis
- **Country/Region Mapping**: Standardized country and region codes
- **Distance Calculations**: Haversine distance and proximity analysis

Geographic Operations:
- Coordinate validation and normalization
- Reverse geocoding with multiple providers
- Location name standardization
- Geographic distance calculations
- Spatial clustering and analysis

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import asyncio
import aiohttp
import logging
import math
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import json

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LocationInfo:
    """Enhanced location information."""
    latitude: float
    longitude: float
    country: Optional[str] = None
    country_code: Optional[str] = None
    admin1: Optional[str] = None  # State/Province
    admin2: Optional[str] = None  # County/District
    city: Optional[str] = None
    place_name: Optional[str] = None
    confidence: float = 0.0
    source: str = "unknown"


@dataclass
class GeographicCluster:
    """Geographic cluster of events."""
    center_lat: float
    center_lon: float
    radius_km: float
    event_count: int
    events: List[str]  # Event IDs
    dominant_location: Optional[str] = None


class CoordinateValidator:
    """Validator for geographic coordinates."""
    
    @staticmethod
    def is_valid_coordinate(lat: float, lon: float) -> bool:
        """Check if coordinates are valid."""
        try:
            return (-90 <= lat <= 90) and (-180 <= lon <= 180)
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def is_likely_valid_location(lat: float, lon: float) -> bool:
        """Check if coordinates represent a likely valid location (not ocean/null island)."""
        # Basic validation
        if not CoordinateValidator.is_valid_coordinate(lat, lon):
            return False
        
        # Check for common invalid coordinates
        # Null Island (0, 0)
        if abs(lat) < 0.1 and abs(lon) < 0.1:
            return False
        
        # Check for obviously invalid patterns
        if lat == lon or lat == -lon:
            return False
        
        return True
    
    @staticmethod
    def normalize_coordinates(lat: float, lon: float) -> Tuple[float, float]:
        """Normalize coordinates to standard precision."""
        # Round to 6 decimal places (approximately 0.1 meter precision)
        lat = round(float(lat), 6)
        lon = round(float(lon), 6)
        
        # Ensure longitude is in -180 to 180 range
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        
        return lat, lon


class DistanceCalculator:
    """Geographic distance calculations."""
    
    EARTH_RADIUS_KM = 6371.0
    
    @classmethod
    def haversine_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return cls.EARTH_RADIUS_KM * c
    
    @classmethod
    def find_nearby_points(
        cls,
        center_lat: float,
        center_lon: float,
        points: List[Tuple[float, float]],
        radius_km: float
    ) -> List[Tuple[float, float, float]]:
        """
        Find points within a radius of a center point.
        
        Args:
            center_lat, center_lon: Center point coordinates
            points: List of (lat, lon) tuples to check
            radius_km: Search radius in kilometers
            
        Returns:
            List of (lat, lon, distance) tuples within radius
        """
        nearby = []
        
        for lat, lon in points:
            distance = cls.haversine_distance(center_lat, center_lon, lat, lon)
            if distance <= radius_km:
                nearby.append((lat, lon, distance))
        
        return sorted(nearby, key=lambda x: x[2])  # Sort by distance


class CountryCodeMapper:
    """Maps between different country code standards."""
    
    # ISO 3166-1 alpha-2 to alpha-3 mapping (subset for common countries)
    ISO2_TO_ISO3 = {
        'AF': 'AFG', 'AL': 'ALB', 'DZ': 'DZA', 'AS': 'ASM', 'AD': 'AND',
        'AO': 'AGO', 'AI': 'AIA', 'AQ': 'ATA', 'AG': 'ATG', 'AR': 'ARG',
        'AM': 'ARM', 'AW': 'ABW', 'AU': 'AUS', 'AT': 'AUT', 'AZ': 'AZE',
        'BS': 'BHS', 'BH': 'BHR', 'BD': 'BGD', 'BB': 'BRB', 'BY': 'BLR',
        'BE': 'BEL', 'BZ': 'BLZ', 'BJ': 'BEN', 'BM': 'BMU', 'BT': 'BTN',
        'BO': 'BOL', 'BA': 'BIH', 'BW': 'BWA', 'BV': 'BVT', 'BR': 'BRA',
        'IO': 'IOT', 'BN': 'BRN', 'BG': 'BGR', 'BF': 'BFA', 'BI': 'BDI',
        'KH': 'KHM', 'CM': 'CMR', 'CA': 'CAN', 'CV': 'CPV', 'KY': 'CYM',
        'CF': 'CAF', 'TD': 'TCD', 'CL': 'CHL', 'CN': 'CHN', 'CX': 'CXR',
        'CC': 'CCK', 'CO': 'COL', 'KM': 'COM', 'CG': 'COG', 'CD': 'COD',
        'CK': 'COK', 'CR': 'CRI', 'CI': 'CIV', 'HR': 'HRV', 'CU': 'CUB',
        'CY': 'CYP', 'CZ': 'CZE', 'DK': 'DNK', 'DJ': 'DJI', 'DM': 'DMA',
        'DO': 'DOM', 'EC': 'ECU', 'EG': 'EGY', 'SV': 'SLV', 'GQ': 'GNQ',
        'ER': 'ERI', 'EE': 'EST', 'ET': 'ETH', 'FK': 'FLK', 'FO': 'FRO',
        'FJ': 'FJI', 'FI': 'FIN', 'FR': 'FRA', 'GF': 'GUF', 'PF': 'PYF',
        'TF': 'ATF', 'GA': 'GAB', 'GM': 'GMB', 'GE': 'GEO', 'DE': 'DEU',
        'GH': 'GHA', 'GI': 'GIB', 'GR': 'GRC', 'GL': 'GRL', 'GD': 'GRD',
        'GP': 'GLP', 'GU': 'GUM', 'GT': 'GTM', 'GG': 'GGY', 'GN': 'GIN',
        'GW': 'GNB', 'GY': 'GUY', 'HT': 'HTI', 'HM': 'HMD', 'VA': 'VAT',
        'HN': 'HND', 'HK': 'HKG', 'HU': 'HUN', 'IS': 'ISL', 'IN': 'IND',
        'ID': 'IDN', 'IR': 'IRN', 'IQ': 'IRQ', 'IE': 'IRL', 'IM': 'IMN',
        'IL': 'ISR', 'IT': 'ITA', 'JM': 'JAM', 'JP': 'JPN', 'JE': 'JEY',
        'JO': 'JOR', 'KZ': 'KAZ', 'KE': 'KEN', 'KI': 'KIR', 'KP': 'PRK',
        'KR': 'KOR', 'KW': 'KWT', 'KG': 'KGZ', 'LA': 'LAO', 'LV': 'LVA',
        'LB': 'LBN', 'LS': 'LSO', 'LR': 'LBR', 'LY': 'LBY', 'LI': 'LIE',
        'LT': 'LTU', 'LU': 'LUX', 'MO': 'MAC', 'MK': 'MKD', 'MG': 'MDG',
        'MW': 'MWI', 'MY': 'MYS', 'MV': 'MDV', 'ML': 'MLI', 'MT': 'MLT',
        'MH': 'MHL', 'MQ': 'MTQ', 'MR': 'MRT', 'MU': 'MUS', 'YT': 'MYT',
        'MX': 'MEX', 'FM': 'FSM', 'MD': 'MDA', 'MC': 'MCO', 'MN': 'MNG',
        'ME': 'MNE', 'MS': 'MSR', 'MA': 'MAR', 'MZ': 'MOZ', 'MM': 'MMR',
        'NA': 'NAM', 'NR': 'NRU', 'NP': 'NPL', 'NL': 'NLD', 'AN': 'ANT',
        'NC': 'NCL', 'NZ': 'NZL', 'NI': 'NIC', 'NE': 'NER', 'NG': 'NGA',
        'NU': 'NIU', 'NF': 'NFK', 'MP': 'MNP', 'NO': 'NOR', 'OM': 'OMN',
        'PK': 'PAK', 'PW': 'PLW', 'PS': 'PSE', 'PA': 'PAN', 'PG': 'PNG',
        'PY': 'PRY', 'PE': 'PER', 'PH': 'PHL', 'PN': 'PCN', 'PL': 'POL',
        'PT': 'PRT', 'PR': 'PRI', 'QA': 'QAT', 'RE': 'REU', 'RO': 'ROU',
        'RU': 'RUS', 'RW': 'RWA', 'BL': 'BLM', 'SH': 'SHN', 'KN': 'KNA',
        'LC': 'LCA', 'MF': 'MAF', 'PM': 'SPM', 'VC': 'VCT', 'WS': 'WSM',
        'SM': 'SMR', 'ST': 'STP', 'SA': 'SAU', 'SN': 'SEN', 'RS': 'SRB',
        'SC': 'SYC', 'SL': 'SLE', 'SG': 'SGP', 'SK': 'SVK', 'SI': 'SVN',
        'SB': 'SLB', 'SO': 'SOM', 'ZA': 'ZAF', 'GS': 'SGS', 'ES': 'ESP',
        'LK': 'LKA', 'SD': 'SDN', 'SR': 'SUR', 'SJ': 'SJM', 'SZ': 'SWZ',
        'SE': 'SWE', 'CH': 'CHE', 'SY': 'SYR', 'TW': 'TWN', 'TJ': 'TJK',
        'TZ': 'TZA', 'TH': 'THA', 'TL': 'TLS', 'TG': 'TGO', 'TK': 'TKL',
        'TO': 'TON', 'TT': 'TTO', 'TN': 'TUN', 'TR': 'TUR', 'TM': 'TKM',
        'TC': 'TCA', 'TV': 'TUV', 'UG': 'UGA', 'UA': 'UKR', 'AE': 'ARE',
        'GB': 'GBR', 'US': 'USA', 'UM': 'UMI', 'UY': 'URY', 'UZ': 'UZB',
        'VU': 'VUT', 'VE': 'VEN', 'VN': 'VNM', 'VG': 'VGB', 'VI': 'VIR',
        'WF': 'WLF', 'EH': 'ESH', 'YE': 'YEM', 'ZM': 'ZMB', 'ZW': 'ZWE'
    }
    
    # Country name standardization
    COUNTRY_ALIASES = {
        'United States': 'United States of America',
        'USA': 'United States of America',
        'UK': 'United Kingdom',
        'Britain': 'United Kingdom',
        'England': 'United Kingdom',
        'Soviet Union': 'Russia',
        'USSR': 'Russia',
        'Congo': 'Democratic Republic of the Congo',
        'Burma': 'Myanmar',
        'Persia': 'Iran'
    }
    
    @classmethod
    def iso2_to_iso3(cls, iso2_code: str) -> Optional[str]:
        """Convert ISO 3166-1 alpha-2 to alpha-3 code."""
        return cls.ISO2_TO_ISO3.get(iso2_code.upper()) if iso2_code else None
    
    @classmethod
    def standardize_country_name(cls, country_name: str) -> str:
        """Standardize country name."""
        if not country_name:
            return ""
        
        # Clean up the name
        cleaned = re.sub(r'\s+', ' ', country_name.strip()).title()
        
        # Apply aliases
        return cls.COUNTRY_ALIASES.get(cleaned, cleaned)


class LocationEnhancer:
    """Enhances location information for GDELT events."""
    
    def __init__(self):
        self.validator = CoordinateValidator()
        self.distance_calc = DistanceCalculator()
        self.country_mapper = CountryCodeMapper()
        
        # Cache for reverse geocoding results
        self._geocoding_cache: Dict[str, LocationInfo] = {}
    
    def enhance_location(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        location_name: Optional[str] = None,
        country_code: Optional[str] = None
    ) -> LocationInfo:
        """
        Enhance location information with validation and standardization.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate  
            location_name: Location name from GDELT
            country_code: Country code from GDELT
            
        Returns:
            Enhanced LocationInfo object
        """
        # Validate and normalize coordinates
        if latitude is not None and longitude is not None:
            if not self.validator.is_valid_coordinate(latitude, longitude):
                logger.warning(f"Invalid coordinates: {latitude}, {longitude}")
                latitude = longitude = None
            else:
                latitude, longitude = self.validator.normalize_coordinates(latitude, longitude)
        
        # Create base location info
        location_info = LocationInfo(
            latitude=latitude or 0.0,
            longitude=longitude or 0.0,
            place_name=self._clean_location_name(location_name),
            country_code=country_code.upper() if country_code else None
        )
        
        # Enhance with country information
        if country_code:
            iso3_code = self.country_mapper.iso2_to_iso3(country_code)
            if iso3_code:
                location_info.country = iso3_code
        
        # Set confidence based on available data
        confidence = 0.0
        if latitude is not None and longitude is not None:
            confidence += 0.5
        if location_name:
            confidence += 0.3
        if country_code:
            confidence += 0.2
        
        location_info.confidence = min(confidence, 1.0)
        location_info.source = "gdelt_enhanced"
        
        return location_info
    
    def _clean_location_name(self, location_name: Optional[str]) -> Optional[str]:
        """Clean and standardize location names."""
        if not location_name:
            return None
        
        # Remove extra whitespace and clean up
        cleaned = re.sub(r'\s+', ' ', location_name.strip())
        
        # Remove common prefixes/suffixes that don't add value
        cleaned = re.sub(r'^(City of|Town of|Village of)\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+(City|Town|Village)$', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned if cleaned else None
    
    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        use_cache: bool = True
    ) -> Optional[LocationInfo]:
        """
        Perform reverse geocoding to get location details from coordinates.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            use_cache: Whether to use cached results
            
        Returns:
            LocationInfo with reverse geocoded data or None
        """
        if not self.validator.is_valid_coordinate(latitude, longitude):
            return None
        
        # Create cache key
        cache_key = f"{latitude:.6f},{longitude:.6f}"
        
        # Check cache first
        if use_cache and cache_key in self._geocoding_cache:
            return self._geocoding_cache[cache_key]
        
        # Try multiple geocoding services (mock implementation)
        location_info = await self._nominatim_reverse_geocode(latitude, longitude)
        
        # Cache result
        if location_info and use_cache:
            self._geocoding_cache[cache_key] = location_info
        
        return location_info
    
    async def _nominatim_reverse_geocode(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[LocationInfo]:
        """
        Reverse geocode using Nominatim (OpenStreetMap) service.
        
        Note: This is a simplified implementation. In production,
        you would want to implement proper rate limiting and error handling.
        """
        try:
            url = f"https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': latitude,
                'lon': longitude,
                'format': 'json',
                'addressdetails': 1,
                'zoom': 10
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_nominatim_response(data, latitude, longitude)
            
        except Exception as e:
            logger.warning(f"Reverse geocoding failed: {e}")
        
        return None
    
    def _parse_nominatim_response(
        self,
        data: Dict[str, Any],
        latitude: float,
        longitude: float
    ) -> LocationInfo:
        """Parse Nominatim response into LocationInfo."""
        address = data.get('address', {})
        
        return LocationInfo(
            latitude=latitude,
            longitude=longitude,
            country=address.get('country'),
            country_code=address.get('country_code', '').upper(),
            admin1=address.get('state') or address.get('province'),
            admin2=address.get('county') or address.get('district'),
            city=address.get('city') or address.get('town') or address.get('village'),
            place_name=data.get('display_name'),
            confidence=0.8,  # Nominatim generally provides good results
            source="nominatim"
        )


class GDELTGeocoder:
    """Main geocoding interface for GDELT data."""
    
    def __init__(self):
        self.enhancer = LocationEnhancer()
        self.distance_calc = DistanceCalculator()
    
    def process_event_location(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and enhance location data for a GDELT event.
        
        Args:
            event_data: GDELT event dictionary
            
        Returns:
            Enhanced event data with location information
        """
        # Extract location data
        latitude = event_data.get('latitude')
        longitude = event_data.get('longitude')
        location_name = event_data.get('location_name')
        country = event_data.get('country')
        
        # Enhance location
        enhanced_location = self.enhancer.enhance_location(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            country_code=country
        )
        
        # Update event data
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
    
    def cluster_events_geographically(
        self,
        events: List[Dict[str, Any]],
        radius_km: float = 50.0,
        min_events: int = 3
    ) -> List[GeographicCluster]:
        """
        Cluster events geographically for hotspot analysis.
        
        Args:
            events: List of GDELT events with coordinates
            radius_km: Clustering radius in kilometers
            min_events: Minimum events to form a cluster
            
        Returns:
            List of geographic clusters
        """
        # Extract valid coordinates
        valid_events = []
        for event in events:
            lat = event.get('latitude')
            lon = event.get('longitude')
            if lat is not None and lon is not None:
                if CoordinateValidator.is_valid_coordinate(lat, lon):
                    valid_events.append(event)
        
        if len(valid_events) < min_events:
            return []
        
        # Simple clustering algorithm
        clusters = []
        used_events = set()
        
        for i, center_event in enumerate(valid_events):
            if i in used_events:
                continue
            
            center_lat = center_event['latitude']
            center_lon = center_event['longitude']
            
            # Find nearby events
            cluster_events = [center_event]
            cluster_event_ids = [center_event.get('external_id', str(i))]
            used_events.add(i)
            
            for j, other_event in enumerate(valid_events):
                if j in used_events:
                    continue
                
                other_lat = other_event['latitude']
                other_lon = other_event['longitude']
                
                distance = self.distance_calc.haversine_distance(
                    center_lat, center_lon, other_lat, other_lon
                )
                
                if distance <= radius_km:
                    cluster_events.append(other_event)
                    cluster_event_ids.append(other_event.get('external_id', str(j)))
                    used_events.add(j)
            
            # Create cluster if it meets minimum size
            if len(cluster_events) >= min_events:
                # Calculate cluster center
                avg_lat = sum(e['latitude'] for e in cluster_events) / len(cluster_events)
                avg_lon = sum(e['longitude'] for e in cluster_events) / len(cluster_events)
                
                # Find dominant location name
                location_names = [e.get('location_name') for e in cluster_events if e.get('location_name')]
                dominant_location = max(set(location_names), key=location_names.count) if location_names else None
                
                cluster = GeographicCluster(
                    center_lat=avg_lat,
                    center_lon=avg_lon,
                    radius_km=radius_km,
                    event_count=len(cluster_events),
                    events=cluster_event_ids,
                    dominant_location=dominant_location
                )
                clusters.append(cluster)
        
        return sorted(clusters, key=lambda c: c.event_count, reverse=True)


# Utility functions
def coordinate_validator(lat: float, lon: float) -> bool:
    """Simple coordinate validation function."""
    return CoordinateValidator.is_valid_coordinate(lat, lon)


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers."""
    return DistanceCalculator.haversine_distance(lat1, lon1, lat2, lon2)


def standardize_location_name(location_name: str) -> str:
    """Standardize a location name."""
    enhancer = LocationEnhancer()
    return enhancer._clean_location_name(location_name) or ""


# Export all components
__all__ = [
    'GDELTGeocoder',
    'LocationEnhancer',
    'LocationInfo',
    'GeographicCluster',
    'CoordinateValidator',
    'DistanceCalculator',
    'CountryCodeMapper',
    'coordinate_validator',
    'calculate_distance',
    'standardize_location_name'
]