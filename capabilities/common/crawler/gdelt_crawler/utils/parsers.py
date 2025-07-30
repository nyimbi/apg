"""
GDELT Format Parsers and Data Transformation Utilities
======================================================

Comprehensive parsing utilities for GDELT data formats with validation,
transformation, and standardization capabilities.

Key Features:
- **Format Validation**: Robust validation of GDELT data formats
- **Event Code Mapping**: CAMEO event code to human-readable descriptions
- **Date/Time Parsing**: GDELT date/time format standardization
- **Coordinate Processing**: Geographic coordinate validation and conversion
- **Data Transformation**: Standardization for database storage

Supported Formats:
- GDELT Events (61-field format)
- GDELT Mentions (16-field format)
- GKG (Global Knowledge Graph)
- Legacy formats and variations

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import json

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ParsedEvent:
    """Represents a parsed GDELT event."""
    event_id: str
    event_date: datetime
    actors: Dict[str, Any]
    event_info: Dict[str, Any]
    location: Dict[str, Any]
    metrics: Dict[str, Any]
    raw_data: Dict[str, Any]


class EventCodeMapper:
    """Maps GDELT CAMEO event codes to human-readable descriptions."""
    
    # CAMEO Event Codes (Conflict and Mediation Event Observations)
    EVENT_CODES = {
        # Verbal Cooperation
        "01": "Make Public Statement",
        "010": "Make statement, not specified below",
        "011": "Decline comment",
        "012": "Make pessimistic comment",
        "013": "Make optimistic comment",
        "014": "Consider policy option",
        "015": "Acknowledge or claim responsibility",
        "016": "Deny responsibility",
        "017": "Engage in symbolic act",
        "018": "Make empathetic comment",
        "019": "Express accord",
        
        "02": "Appeal",
        "020": "Appeal, not specified below",
        "021": "Appeal for material cooperation",
        "022": "Appeal for diplomatic cooperation",
        "023": "Appeal for aid",
        "024": "Appeal for political reform",
        "025": "Appeal for change in leadership",
        "026": "Appeal for policy change",
        "027": "Appeal for rights",
        "028": "Appeal for peaceful settlement",
        
        "03": "Express Intent to Cooperate",
        "030": "Express intent to cooperate, not specified below",
        "031": "Express intent to engage in material cooperation",
        "032": "Express intent to engage in diplomatic cooperation",
        "033": "Express intent to provide aid",
        "034": "Express intent to institute political reform",
        "035": "Express intent to change leadership",
        "036": "Express intent to change policy",
        "037": "Express intent to provide rights",
        "038": "Express intent to settle dispute peacefully",
        "039": "Express intent to cooperate economically",
        
        "04": "Consult",
        "040": "Consult, not specified below",
        "041": "Discuss by telephone",
        "042": "Make a visit",
        "043": "Host a visit",
        "044": "Meet at a third location",
        "045": "Mediate",
        "046": "Engage in negotiation",
        
        "05": "Engage in Diplomatic Cooperation",
        "050": "Engage in diplomatic cooperation, not specified below",
        "051": "Praise or endorse",
        "052": "Defend verbally",
        "053": "Rally support on behalf of",
        "054": "Grant diplomatic recognition",
        "055": "Apologize",
        "056": "Forgive",
        "057": "Sign formal agreement",
        
        "06": "Engage in Material Cooperation",
        "060": "Engage in material cooperation, not specified below",
        "061": "Cooperate economically",
        "062": "Cooperate militarily",
        "063": "Engage in judicial cooperation",
        "064": "Share intelligence or information",
        
        "07": "Provide Aid",
        "070": "Provide aid, not specified below",
        "071": "Provide economic aid",
        "072": "Provide military aid",
        "073": "Provide humanitarian aid",
        "074": "Provide military protection or peacekeeping",
        "075": "Grant asylum",
        
        "08": "Yield",
        "080": "Yield, not specified below",
        "081": "Ease administrative sanctions",
        "082": "Ease political dissent",
        "083": "Accede to requests or demands",
        "084": "Return, release, or allow",
        "085": "Ease restrictions on political freedoms",
        "086": "Release persons or property",
        
        "09": "Investigate",
        "090": "Investigate, not specified below",
        "091": "Investigate crime, corruption",
        "092": "Investigate human rights abuses",
        "093": "Investigate military action",
        "094": "Investigate war crimes",
        
        # Verbal Conflict
        "10": "Demand",
        "100": "Demand, not specified below",
        "101": "Demand material cooperation",
        "102": "Demand diplomatic cooperation",
        "103": "Demand aid",
        "104": "Demand political reform",
        "105": "Demand change in leadership",
        "106": "Demand policy change",
        "107": "Demand rights",
        "108": "Demand that target yields",
        "109": "Demand de-escalation of military engagement",
        
        "11": "Disapprove",
        "110": "Disapprove, not specified below",
        "111": "Criticize or denounce",
        "112": "Accuse",
        "113": "Rally opposition against",
        "114": "Complain officially",
        "115": "Bring lawsuit against",
        "116": "Find guilty or liable",
        
        "12": "Reject",
        "120": "Reject, not specified below",
        "121": "Reject material cooperation",
        "122": "Reject diplomatic cooperation",
        "123": "Reject aid",
        "124": "Reject political reform",
        "125": "Reject change in leadership",
        "126": "Reject policy change",
        "127": "Reject rights",
        "128": "Reject plan, agreement, or settlement",
        "129": "Reject de-escalation of military engagement",
        
        "13": "Threaten",
        "130": "Threaten, not specified below",
        "131": "Threaten non-force",
        "132": "Threaten to reduce or stop aid",
        "133": "Threaten to reduce or break relations",
        "134": "Threaten with administrative sanctions",
        "135": "Threaten with political dissent, protest",
        "136": "Threaten to occupy territory",
        "137": "Threaten with unconventional violence",
        "138": "Threaten with conventional attack",
        "139": "Give ultimatum",
        
        "14": "Protest",
        "140": "Protest, not specified below",
        "141": "Demonstrate or rally",
        "142": "Conduct hunger strike",
        "143": "Conduct strike or boycott",
        "144": "Obstruct passage, block",
        "145": "Protest violently, riot",
        
        "15": "Exhibit Force Posture",
        "150": "Exhibit force posture, not specified below",
        "151": "Increase police alert status",
        "152": "Increase military alert status",
        "153": "Mobilize or increase police power",
        "154": "Mobilize or increase armed forces",
        
        # Material Conflict
        "16": "Reduce Relations",
        "160": "Reduce relations, not specified below",
        "161": "Reduce or stop aid",
        "162": "Break diplomatic relations",
        "163": "Impose administrative sanctions",
        "164": "Impose restrictions on political freedoms",
        "165": "Impose judicial punishment",
        "166": "Expel or deport individuals",
        "167": "Expel or withdraw",
        "168": "Impose restrictions on movement",
        "169": "Impose blockade, restrict movement",
        
        "17": "Coerce",
        "170": "Coerce, not specified below",
        "171": "Seize or damage property",
        "172": "Impose administrative sanctions",
        "173": "Arrest, detain, or charge with legal action",
        "174": "Expel or deport individuals",
        "175": "Use tactics of violent repression",
        
        "18": "Assault",
        "180": "Assault, not specified below",
        "181": "Abduct, hijack, or take hostage",
        "182": "Physically assault",
        "183": "Conduct suicide, car, or other non-military bombing",
        "184": "Use as human shield",
        "185": "Attempt to assassinate",
        "186": "Assassinate",
        
        "19": "Fight",
        "190": "Fight, not specified below",
        "191": "Impose blockade, restrict movement",
        "192": "Occupy territory",
        "193": "Fight with small arms and light weapons",
        "194": "Fight with artillery and tanks",
        "195": "Employ aerial weapons",
        "196": "Violate ceasefire",
        
        "20": "Use Unconventional Mass Violence",
        "200": "Use unconventional mass violence, not specified below",
        "201": "Engage in mass expulsion",
        "202": "Engage in mass killing",
        "203": "Engage in ethnic cleansing",
        "204": "Use weapons of mass destruction"
    }
    
    @classmethod
    def get_description(cls, code: str) -> str:
        """Get human-readable description for an event code."""
        if not code:
            return "Unknown Event"
        
        # Try exact match first
        if code in cls.EVENT_CODES:
            return cls.EVENT_CODES[code]
        
        # Try base code (first 2 digits)
        if len(code) >= 2:
            base_code = code[:2]
            if base_code in cls.EVENT_CODES:
                return cls.EVENT_CODES[base_code]
        
        return f"Event Code {code}"
    
    @classmethod
    def get_category(cls, code: str) -> str:
        """Get event category from code."""
        if not code:
            return "Unknown"
        
        try:
            base_code = int(code[:2]) if len(code) >= 2 else 0
            
            if 1 <= base_code <= 9:
                return "Cooperation"
            elif 10 <= base_code <= 15:
                return "Verbal Conflict"
            elif 16 <= base_code <= 20:
                return "Material Conflict"
            else:
                return "Other"
        except (ValueError, IndexError):
            return "Unknown"
    
    @classmethod
    def is_conflict_event(cls, code: str) -> bool:
        """Check if event code represents a conflict."""
        try:
            base_code = int(code[:2]) if code and len(code) >= 2 else 0
            return base_code >= 10  # Codes 10+ are conflict-related
        except (ValueError, IndexError):
            return False


class GDELTFormatParser:
    """Comprehensive parser for GDELT data formats."""
    
    def __init__(self):
        self.event_mapper = EventCodeMapper()
    
    def parse_events_record(self, fields: List[str]) -> ParsedEvent:
        """Parse a GDELT Events record (61-field format)."""
        if len(fields) < 61:
            raise ValueError(f"Events record must have 61 fields, got {len(fields)}")
        
        try:
            # Basic event information
            event_id = fields[0] or ""
            event_date = self._parse_event_date(fields[1])
            
            # Actors
            actors = {
                "actor1": {
                    "code": fields[5] or None,
                    "name": fields[6] or None,
                    "country": fields[7] or None,
                    "known_group": fields[8] or None,
                    "ethnic": fields[9] or None,
                    "religion1": fields[10] or None,
                    "religion2": fields[11] or None,
                    "type1": fields[12] or None,
                    "type2": fields[13] or None,
                    "type3": fields[14] or None
                },
                "actor2": {
                    "code": fields[15] or None,
                    "name": fields[16] or None,
                    "country": fields[17] or None,
                    "known_group": fields[18] or None,
                    "ethnic": fields[19] or None,
                    "religion1": fields[20] or None,
                    "religion2": fields[21] or None,
                    "type1": fields[22] or None,
                    "type2": fields[23] or None,
                    "type3": fields[24] or None
                }
            }
            
            # Event information
            event_code = fields[26] or ""
            event_info = {
                "is_root_event": self._parse_bool(fields[25]),
                "event_code": event_code,
                "event_description": self.event_mapper.get_description(event_code),
                "event_category": self.event_mapper.get_category(event_code),
                "is_conflict": self.event_mapper.is_conflict_event(event_code),
                "base_code": fields[27] or None,
                "root_code": fields[28] or None,
                "quad_class": self._parse_int(fields[29])
            }
            
            # Metrics
            metrics = {
                "goldstein_scale": self._parse_float(fields[30]),
                "num_mentions": self._parse_int(fields[31]),
                "num_sources": self._parse_int(fields[32]),
                "num_articles": self._parse_int(fields[33]),
                "avg_tone": self._parse_float(fields[34])
            }
            
            # Geographic information
            location = {
                "actor1_geo": self._parse_geographic_fields(fields[35:45]),
                "actor2_geo": self._parse_geographic_fields(fields[45:55]),
                "action_geo": self._parse_geographic_fields(fields[55:65] if len(fields) >= 65 else fields[55:61])
            }
            
            # Additional fields
            raw_data = {
                "date_added": fields[59] if len(fields) > 59 else None,
                "source_url": fields[60] if len(fields) > 60 else None
            }
            
            return ParsedEvent(
                event_id=event_id,
                event_date=event_date,
                actors=actors,
                event_info=event_info,
                location=location,
                metrics=metrics,
                raw_data=raw_data
            )
            
        except Exception as e:
            logger.error(f"Failed to parse events record: {e}")
            raise ValueError(f"Invalid events record format: {e}")
    
    def _parse_geographic_fields(self, geo_fields: List[str]) -> Dict[str, Any]:
        """Parse geographic fields (Type, FullName, CountryCode, ADM1Code, ADM2Code, Lat, Long, FeatureID)."""
        if len(geo_fields) < 8:
            return {}
        
        return {
            "type": self._parse_int(geo_fields[0]),
            "full_name": geo_fields[1] or None,
            "country_code": geo_fields[2] or None,
            "adm1_code": geo_fields[3] or None,
            "adm2_code": geo_fields[4] or None,
            "latitude": self._parse_float(geo_fields[5]),
            "longitude": self._parse_float(geo_fields[6]),
            "feature_id": geo_fields[7] or None
        }
    
    def _parse_event_date(self, date_str: str) -> Optional[datetime]:
        """Parse GDELT event date (YYYYMMDD format)."""
        if not date_str or len(date_str) != 8:
            return None
        
        try:
            return datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    
    def _parse_bool(self, value: str) -> Optional[bool]:
        """Parse boolean value."""
        if not value:
            return None
        return value == "1"
    
    def _parse_int(self, value: str) -> Optional[int]:
        """Parse integer value."""
        if not value or value.strip() == "":
            return None
        try:
            return int(float(value))  # Handle decimal integers
        except ValueError:
            return None
    
    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value."""
        if not value or value.strip() == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None


# Utility functions
def parse_gdelt_date(date_str: str, format_type: str = "date") -> Optional[datetime]:
    """
    Parse GDELT date/datetime strings.
    
    Args:
        date_str: Date string to parse
        format_type: Type of format ('date', 'datetime', 'timestamp')
        
    Returns:
        Parsed datetime object or None
    """
    if not date_str:
        return None
    
    try:
        if format_type == "date" and len(date_str) == 8:
            # YYYYMMDD format
            return datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        elif format_type == "datetime" and len(date_str) == 14:
            # YYYYMMDDHHMMSS format
            return datetime.strptime(date_str, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        elif format_type == "timestamp":
            # Unix timestamp
            return datetime.fromtimestamp(float(date_str), tz=timezone.utc)
        else:
            # Try to parse as ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


def parse_gdelt_coordinates(lat_str: str, lon_str: str) -> Optional[Tuple[float, float]]:
    """
    Parse and validate GDELT coordinate strings.
    
    Args:
        lat_str: Latitude string
        lon_str: Longitude string
        
    Returns:
        Tuple of (latitude, longitude) or None if invalid
    """
    try:
        lat = float(lat_str) if lat_str else None
        lon = float(lon_str) if lon_str else None
        
        if lat is None or lon is None:
            return None
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None
        
        return (lat, lon)
    except (ValueError, TypeError):
        return None


def normalize_actor_name(actor_name: str) -> str:
    """
    Normalize actor names for consistency.
    
    Args:
        actor_name: Raw actor name
        
    Returns:
        Normalized actor name
    """
    if not actor_name:
        return ""
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', actor_name.strip())
    
    # Convert to title case for consistency
    normalized = normalized.title()
    
    # Handle common abbreviations
    abbreviations = {
        'U.s.': 'U.S.',
        'U.k.': 'U.K.',
        'U.n.': 'U.N.',
        'Nato': 'NATO',
        'Eu': 'EU'
    }
    
    for old, new in abbreviations.items():
        normalized = normalized.replace(old, new)
    
    return normalized


def extract_themes_from_gkg(themes_str: str) -> List[Dict[str, Any]]:
    """
    Extract and parse themes from GKG themes field.
    
    Args:
        themes_str: GKG themes string
        
    Returns:
        List of theme dictionaries
    """
    if not themes_str:
        return []
    
    themes = []
    for theme in themes_str.split(';'):
        if theme.strip():
            parts = theme.split(',')
            if len(parts) >= 2:
                themes.append({
                    'name': parts[0].strip(),
                    'score': float(parts[1]) if parts[1].replace('.', '').isdigit() else None,
                    'raw': theme
                })
    
    return themes


def validate_gdelt_record(record: Dict[str, Any], dataset_type: str) -> List[str]:
    """
    Validate a GDELT record for completeness and correctness.
    
    Args:
        record: GDELT record dictionary
        dataset_type: Type of dataset ('events', 'mentions', 'gkg')
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Common validations
    if not record.get('external_id'):
        errors.append("Missing external_id")
    
    # Dataset-specific validations
    if dataset_type == 'events':
        if not record.get('event_date'):
            errors.append("Missing event_date")
        
        # Validate coordinates if present
        lat = record.get('latitude')
        lon = record.get('longitude')
        if lat is not None and lon is not None:
            if not (-90 <= lat <= 90):
                errors.append(f"Invalid latitude: {lat}")
            if not (-180 <= lon <= 180):
                errors.append(f"Invalid longitude: {lon}")
    
    elif dataset_type == 'mentions':
        if not record.get('mention_time_date'):
            errors.append("Missing mention_time_date")
    
    elif dataset_type == 'gkg':
        if not record.get('date'):
            errors.append("Missing date")
    
    return errors


# Export all functions
__all__ = [
    'GDELTFormatParser',
    'EventCodeMapper', 
    'ParsedEvent',
    'parse_gdelt_date',
    'parse_gdelt_coordinates',
    'normalize_actor_name',
    'extract_themes_from_gkg',
    'validate_gdelt_record'
]