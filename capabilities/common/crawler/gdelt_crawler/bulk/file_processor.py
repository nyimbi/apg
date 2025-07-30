"""
GDELT File Processor with Advanced Data Parsing and Validation
==============================================================

A comprehensive processor for GDELT CSV files with field validation, data transformation,
and database-ready formatting. Supports all GDELT dataset formats (Events, Mentions, GKG)
with robust error handling and performance optimization.

Key Features:
- **Multi-Format Support**: Events, Mentions, and GKG dataset parsing
- **Field Validation**: Comprehensive validation of all GDELT fields
- **Data Transformation**: Automatic type conversion and normalization
- **Error Recovery**: Robust handling of malformed records
- **Performance Optimization**: Streaming processing for large files
- **Schema Mapping**: Direct mapping to information_units database schema
- **Content Extraction**: ML-ready content preparation
- **Batch Processing**: Configurable batch sizes for memory efficiency

Supported GDELT Formats:
- **Events**: Core event database with 61+ fields
- **Mentions**: Event mentions in global media
- **GKG**: Global Knowledge Graph with themes and entities
- **Export**: Simplified event export format

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import asyncio
import aiofiles
import csv
import logging
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator, AsyncIterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import gzip
import zipfile
import io
from collections import defaultdict
import hashlib

# Configure logging
logger = logging.getLogger(__name__)


class GDELTFieldType(Enum):
    """GDELT field data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    JSON = "json"
    LIST = "list"


@dataclass
class GDELTField:
    """GDELT field definition."""
    name: str
    gdelt_name: str
    field_type: GDELTFieldType
    required: bool = False
    default: Any = None
    validator: Optional[callable] = None
    transformer: Optional[callable] = None
    description: str = ""


@dataclass
class ProcessingStats:
    """Processing statistics."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    file_size_bytes: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_records == 0:
            return 100.0
        return (self.valid_records / self.total_records) * 100.0


class GDELTFileProcessor:
    """
    Processes GDELT CSV files and converts them to database-ready format.
    
    Handles all GDELT dataset types with comprehensive field validation,
    error recovery, and performance optimization for large files.
    """
    
    # GDELT Events dataset field mapping (61 fields)
    EVENTS_FIELDS = [
        GDELTField("external_id", "GLOBALEVENTID", GDELTFieldType.STRING, required=True),
        GDELTField("event_date", "SQLDATE", GDELTFieldType.DATE, required=True),
        GDELTField("event_month", "MonthYear", GDELTFieldType.INTEGER),
        GDELTField("event_year", "Year", GDELTFieldType.INTEGER),
        GDELTField("event_fraction_date", "FractionDate", GDELTFieldType.FLOAT),
        
        # Actor 1
        GDELTField("actor1_code", "Actor1Code", GDELTFieldType.STRING),
        GDELTField("actor1_name", "Actor1Name", GDELTFieldType.STRING),
        GDELTField("actor1_country_code", "Actor1CountryCode", GDELTFieldType.STRING),
        GDELTField("actor1_known_group_code", "Actor1KnownGroupCode", GDELTFieldType.STRING),
        GDELTField("actor1_ethnic_code", "Actor1EthnicCode", GDELTFieldType.STRING),
        GDELTField("actor1_religion1_code", "Actor1Religion1Code", GDELTFieldType.STRING),
        GDELTField("actor1_religion2_code", "Actor1Religion2Code", GDELTFieldType.STRING),
        GDELTField("actor1_type1_code", "Actor1Type1Code", GDELTFieldType.STRING),
        GDELTField("actor1_type2_code", "Actor1Type2Code", GDELTFieldType.STRING),
        GDELTField("actor1_type3_code", "Actor1Type3Code", GDELTFieldType.STRING),
        
        # Actor 2
        GDELTField("actor2_code", "Actor2Code", GDELTFieldType.STRING),
        GDELTField("actor2_name", "Actor2Name", GDELTFieldType.STRING),
        GDELTField("actor2_country_code", "Actor2CountryCode", GDELTFieldType.STRING),
        GDELTField("actor2_known_group_code", "Actor2KnownGroupCode", GDELTFieldType.STRING),
        GDELTField("actor2_ethnic_code", "Actor2EthnicCode", GDELTFieldType.STRING),
        GDELTField("actor2_religion1_code", "Actor2Religion1Code", GDELTFieldType.STRING),
        GDELTField("actor2_religion2_code", "Actor2Religion2Code", GDELTFieldType.STRING),
        GDELTField("actor2_type1_code", "Actor2Type1Code", GDELTFieldType.STRING),
        GDELTField("actor2_type2_code", "Actor2Type2Code", GDELTFieldType.STRING),
        GDELTField("actor2_type3_code", "Actor2Type3Code", GDELTFieldType.STRING),
        
        # Event
        GDELTField("is_root_event", "IsRootEvent", GDELTFieldType.BOOLEAN),
        GDELTField("event_id_cnty", "EventIdCnty", GDELTFieldType.STRING),
        GDELTField("event_id_known_group", "EventIdKnownGroup", GDELTFieldType.STRING),
        GDELTField("event_code", "EventCode", GDELTFieldType.STRING),
        GDELTField("event_base_code", "EventBaseCode", GDELTFieldType.STRING),
        GDELTField("event_root_code", "EventRootCode", GDELTFieldType.STRING),
        GDELTField("quad_class", "QuadClass", GDELTFieldType.INTEGER),
        GDELTField("goldstein_scale", "GoldsteinScale", GDELTFieldType.FLOAT),
        GDELTField("num_mentions", "NumMentions", GDELTFieldType.INTEGER),
        GDELTField("num_sources", "NumSources", GDELTFieldType.INTEGER),
        GDELTField("num_articles", "NumArticles", GDELTFieldType.INTEGER),
        GDELTField("avg_tone", "AvgTone", GDELTFieldType.FLOAT),
        
        # Actor 1 Geography
        GDELTField("actor1_geo_type", "Actor1Geo_Type", GDELTFieldType.INTEGER),
        GDELTField("actor1_geo_fullname", "Actor1Geo_FullName", GDELTFieldType.STRING),
        GDELTField("actor1_geo_country_code", "Actor1Geo_CountryCode", GDELTFieldType.STRING),
        GDELTField("actor1_geo_adm1_code", "Actor1Geo_ADM1Code", GDELTFieldType.STRING),
        GDELTField("actor1_geo_adm2_code", "Actor1Geo_ADM2Code", GDELTFieldType.STRING),
        GDELTField("actor1_geo_lat", "Actor1Geo_Lat", GDELTFieldType.FLOAT),
        GDELTField("actor1_geo_long", "Actor1Geo_Long", GDELTFieldType.FLOAT),
        GDELTField("actor1_geo_feature_id", "Actor1Geo_FeatureID", GDELTFieldType.STRING),
        
        # Actor 2 Geography
        GDELTField("actor2_geo_type", "Actor2Geo_Type", GDELTFieldType.INTEGER),
        GDELTField("actor2_geo_fullname", "Actor2Geo_FullName", GDELTFieldType.STRING),
        GDELTField("actor2_geo_country_code", "Actor2Geo_CountryCode", GDELTFieldType.STRING),
        GDELTField("actor2_geo_adm1_code", "Actor2Geo_ADM1Code", GDELTFieldType.STRING),
        GDELTField("actor2_geo_adm2_code", "Actor2Geo_ADM2Code", GDELTFieldType.STRING),
        GDELTField("actor2_geo_lat", "Actor2Geo_Lat", GDELTFieldType.FLOAT),
        GDELTField("actor2_geo_long", "Actor2Geo_Long", GDELTFieldType.FLOAT),
        GDELTField("actor2_geo_feature_id", "Actor2Geo_FeatureID", GDELTFieldType.STRING),
        
        # Action Geography
        GDELTField("action_geo_type", "ActionGeo_Type", GDELTFieldType.INTEGER),
        GDELTField("location_name", "ActionGeo_FullName", GDELTFieldType.STRING),
        GDELTField("country", "ActionGeo_CountryCode", GDELTFieldType.STRING),
        GDELTField("action_geo_adm1_code", "ActionGeo_ADM1Code", GDELTFieldType.STRING),
        GDELTField("action_geo_adm2_code", "ActionGeo_ADM2Code", GDELTFieldType.STRING),
        GDELTField("latitude", "ActionGeo_Lat", GDELTFieldType.FLOAT),
        GDELTField("longitude", "ActionGeo_Long", GDELTFieldType.FLOAT),
        GDELTField("action_geo_feature_id", "ActionGeo_FeatureID", GDELTFieldType.STRING),
        
        # Date Added
        GDELTField("date_added", "DATEADDED", GDELTFieldType.DATETIME),
        GDELTField("source_url", "SOURCEURL", GDELTFieldType.STRING),
    ]
    
    # GDELT Mentions dataset field mapping
    MENTIONS_FIELDS = [
        GDELTField("external_id", "GLOBALEVENTID", GDELTFieldType.STRING, required=True),
        GDELTField("event_time_date", "EventTimeDate", GDELTFieldType.DATETIME),
        GDELTField("mention_time_date", "MentionTimeDate", GDELTFieldType.DATETIME),
        GDELTField("mention_type", "MentionType", GDELTFieldType.INTEGER),
        GDELTField("mention_source_name", "MentionSourceName", GDELTFieldType.STRING),
        GDELTField("mention_identifier", "MentionIdentifier", GDELTFieldType.STRING),
        GDELTField("sentence_id", "SentenceID", GDELTFieldType.INTEGER),
        GDELTField("actor1_char_offset", "Actor1CharOffset", GDELTFieldType.INTEGER),
        GDELTField("actor2_char_offset", "Actor2CharOffset", GDELTFieldType.INTEGER),
        GDELTField("action_char_offset", "ActionCharOffset", GDELTFieldType.INTEGER),
        GDELTField("in_raw_text", "InRawText", GDELTFieldType.BOOLEAN),
        GDELTField("confidence", "Confidence", GDELTFieldType.INTEGER),
        GDELTField("mention_doc_len", "MentionDocLen", GDELTFieldType.INTEGER),
        GDELTField("mention_doc_tone", "MentionDocTone", GDELTFieldType.FLOAT),
        GDELTField("mention_doc_translation_info", "MentionDocTranslationInfo", GDELTFieldType.STRING),
        GDELTField("extras", "Extras", GDELTFieldType.STRING),
    ]
    
    # GDELT GKG (Global Knowledge Graph) field mapping - simplified key fields
    GKG_FIELDS = [
        GDELTField("gkg_record_id", "GKGRECORDID", GDELTFieldType.STRING, required=True),
        GDELTField("date", "DATE", GDELTFieldType.DATETIME, required=True),
        GDELTField("source_collection_identifier", "SourceCollectionIdentifier", GDELTFieldType.INTEGER),
        GDELTField("source_common_name", "SourceCommonName", GDELTFieldType.STRING),
        GDELTField("document_identifier", "DocumentIdentifier", GDELTFieldType.STRING),
        GDELTField("counts", "Counts", GDELTFieldType.STRING),
        GDELTField("themes", "Themes", GDELTFieldType.STRING),
        GDELTField("locations", "Locations", GDELTFieldType.STRING),
        GDELTField("persons", "Persons", GDELTFieldType.STRING),
        GDELTField("organizations", "Organizations", GDELTFieldType.STRING),
        GDELTField("tone", "Tone", GDELTFieldType.STRING),
        GDELTField("dates", "Dates", GDELTFieldType.STRING),
        GDELTField("gcam", "GCAM", GDELTFieldType.STRING),
        GDELTField("sharing_image", "SharingImage", GDELTFieldType.STRING),
        GDELTField("related_images", "RelatedImages", GDELTFieldType.STRING),
        GDELTField("social_image_embeds", "SocialImageEmbeds", GDELTFieldType.STRING),
        GDELTField("social_video_embeds", "SocialVideoEmbeds", GDELTFieldType.STRING),
        GDELTField("quotations", "Quotations", GDELTFieldType.STRING),
        GDELTField("all_names", "AllNames", GDELTFieldType.STRING),
        GDELTField("amounts", "Amounts", GDELTFieldType.STRING),
        GDELTField("translation_info", "TranslationInfo", GDELTFieldType.STRING),
        GDELTField("extras", "Extras", GDELTFieldType.STRING),
    ]
    
    def __init__(self):
        self.field_mappings = {
            "events": {field.gdelt_name: field for field in self.EVENTS_FIELDS},
            "mentions": {field.gdelt_name: field for field in self.MENTIONS_FIELDS},
            "gkg": {field.gdelt_name: field for field in self.GKG_FIELDS},
        }
        
        # Create index mappings for each dataset
        self.field_indices = {
            dataset: {i: field for i, field in enumerate(fields.values())}
            for dataset, fields in self.field_mappings.items()
        }
    
    async def process_file(
        self,
        file_path: Path,
        dataset_type: str,
        batch_size: int = 1000,
        validate_records: bool = True
    ) -> AsyncIterator[Tuple[List[Dict[str, Any]], ProcessingStats]]:
        """
        Process a GDELT file and yield batches of database-ready records.
        
        Args:
            file_path: Path to the GDELT file
            dataset_type: Type of dataset ('events', 'mentions', 'gkg')
            batch_size: Number of records per batch
            validate_records: Whether to validate records
            
        Yields:
            Tuples of (batch_records, processing_stats)
        """
        dataset_type = dataset_type.lower()
        if dataset_type not in self.field_mappings:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        stats = ProcessingStats()
        stats.file_size_bytes = file_path.stat().st_size
        
        start_time = datetime.now()
        batch_records = []
        
        try:
            async for raw_record in self._read_file(file_path):
                stats.total_records += 1
                
                try:
                    # Parse and validate record
                    parsed_record = self._parse_record(raw_record, dataset_type)
                    
                    if validate_records:
                        validated_record = self._validate_record(parsed_record, dataset_type)
                    else:
                        validated_record = parsed_record
                    
                    # Transform to database format
                    db_record = self._transform_to_db_format(validated_record, dataset_type)
                    
                    batch_records.append(db_record)
                    stats.valid_records += 1
                    
                except Exception as e:
                    stats.invalid_records += 1
                    error_msg = f"Record {stats.total_records}: {str(e)}"
                    stats.errors.append(error_msg)
                    logger.warning(error_msg)
                    continue
                
                # Yield batch when full
                if len(batch_records) >= batch_size:
                    yield batch_records, stats
                    batch_records = []
            
            # Yield remaining records
            if batch_records:
                yield batch_records, stats
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise
        
        finally:
            stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()
    
    async def process_file_complete(
        self,
        file_path: Path,
        dataset_type: str,
        validate_records: bool = True
    ) -> Tuple[List[Dict[str, Any]], ProcessingStats]:
        """
        Process entire file and return all records with statistics.
        
        Args:
            file_path: Path to the GDELT file
            dataset_type: Type of dataset
            validate_records: Whether to validate records
            
        Returns:
            Tuple of (all_records, processing_stats)
        """
        all_records = []
        final_stats = None
        
        async for batch_records, stats in self.process_file(
            file_path, dataset_type, batch_size=10000, validate_records=validate_records
        ):
            all_records.extend(batch_records)
            final_stats = stats
        
        return all_records, final_stats or ProcessingStats()
    
    async def _read_file(self, file_path: Path) -> AsyncIterator[List[str]]:
        """Read file and yield CSV records."""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.gz':
            # Handle gzipped files
            async for record in self._read_gzipped_file(file_path):
                yield record
        elif file_extension == '.zip':
            # Handle zipped files
            async for record in self._read_zipped_file(file_path):
                yield record
        else:
            # Handle regular CSV files
            async for record in self._read_csv_file(file_path):
                yield record
    
    async def _read_csv_file(self, file_path: Path) -> AsyncIterator[List[str]]:
        """Read regular CSV file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
            
            # Use StringIO for CSV parsing
            csv_reader = csv.reader(io.StringIO(content), delimiter='\t')
            
            for row in csv_reader:
                yield row
    
    async def _read_gzipped_file(self, file_path: Path) -> AsyncIterator[List[str]]:
        """Read gzipped CSV file."""
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            
            for row in csv_reader:
                yield row
    
    async def _read_zipped_file(self, file_path: Path) -> AsyncIterator[List[str]]:
        """Read zipped CSV file."""
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Get the first CSV file in the zip
            csv_files = [name for name in zip_ref.namelist() if name.endswith('.csv')]
            
            if not csv_files:
                raise ValueError(f"No CSV files found in {file_path}")
            
            with zip_ref.open(csv_files[0]) as csv_file:
                # Decode and parse CSV
                content = csv_file.read().decode('utf-8', errors='replace')
                csv_reader = csv.reader(io.StringIO(content), delimiter='\t')
                
                for row in csv_reader:
                    yield row
    
    def _parse_record(self, raw_record: List[str], dataset_type: str) -> Dict[str, Any]:
        """Parse a raw CSV record into a dictionary."""
        field_indices = self.field_indices[dataset_type]
        parsed_record = {}
        
        for i, value in enumerate(raw_record):
            if i in field_indices:
                field = field_indices[i]
                parsed_record[field.name] = self._convert_field_value(value, field)
        
        return parsed_record
    
    def _convert_field_value(self, value: str, field: GDELTField) -> Any:
        """Convert a string value to the appropriate data type."""
        if not value or value.strip() == "":
            return field.default
        
        value = value.strip()
        
        try:
            if field.field_type == GDELTFieldType.STRING:
                return value
            elif field.field_type == GDELTFieldType.INTEGER:
                return int(float(value)) if value else field.default
            elif field.field_type == GDELTFieldType.FLOAT:
                return float(value) if value else field.default
            elif field.field_type == GDELTFieldType.BOOLEAN:
                return value.lower() in ('1', 'true', 't', 'yes', 'y')
            elif field.field_type == GDELTFieldType.DATE:
                return self._parse_date(value)
            elif field.field_type == GDELTFieldType.DATETIME:
                return self._parse_datetime(value)
            elif field.field_type == GDELTFieldType.JSON:
                return json.loads(value) if value else {}
            elif field.field_type == GDELTFieldType.LIST:
                return value.split(';') if value else []
            else:
                return value
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert {value} for field {field.name}: {e}")
            return field.default
    
    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse GDELT date format."""
        if not value:
            return None
        
        try:
            # GDELT date format is typically YYYYMMDD
            if len(value) == 8 and value.isdigit():
                return datetime.strptime(value, '%Y%m%d').replace(tzinfo=timezone.utc)
            else:
                # Try other common formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
                    try:
                        return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
        except ValueError:
            pass
        
        return None
    
    def _parse_datetime(self, value: str) -> Optional[datetime]:
        """Parse GDELT datetime format."""
        if not value:
            return None
        
        try:
            # GDELT datetime format is typically YYYYMMDDHHMMSS
            if len(value) == 14 and value.isdigit():
                return datetime.strptime(value, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
            elif len(value) == 8 and value.isdigit():
                # Date only, add midnight time
                return datetime.strptime(value, '%Y%m%d').replace(tzinfo=timezone.utc)
            else:
                # Try ISO format
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        return None
    
    def _validate_record(self, record: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
        """Validate a parsed record."""
        field_mapping = self.field_mappings[dataset_type]
        validated_record = {}
        
        for field_name, field_def in field_mapping.items():
            value = record.get(field_def.name)
            
            # Check required fields
            if field_def.required and (value is None or value == ""):
                raise ValueError(f"Required field {field_def.name} is missing or empty")
            
            # Apply custom validator if provided
            if field_def.validator and value is not None:
                if not field_def.validator(value):
                    raise ValueError(f"Validation failed for field {field_def.name}: {value}")
            
            validated_record[field_def.name] = value
        
        return validated_record
    
    def _transform_to_db_format(self, record: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
        """Transform record to database format compatible with information_units table."""
        # Base database record
        db_record = {
            'external_id': record.get('external_id', ''),
            'data_source_id': self._get_data_source_id(),
            'title': self._generate_title(record, dataset_type),
            'content_url': record.get('source_url', ''),
            'content': self._generate_content(record, dataset_type),
            'content_hash': self._generate_content_hash(record),
            'language_code': 'en',  # GDELT is primarily English
            'summary': self._generate_summary(record, dataset_type),
            'keywords': self._extract_keywords(record, dataset_type),
            'tags': self._extract_tags(record, dataset_type),
            'authors': [],
            'unit_type': 'gdelt_' + dataset_type,
            'published_at': record.get('event_date') or record.get('date'),
            
            # ML Scoring fields
            'relevance_score': None,
            'confidence_score': record.get('confidence', 0) / 100.0 if record.get('confidence') else None,
            'sentiment_score': self._calculate_sentiment_score(record),
            'verification_status': 'unverified',
            'metadata': self._build_metadata(record, dataset_type),
        }
        
        # Add dataset-specific fields
        if dataset_type == 'events':
            db_record.update(self._transform_events_fields(record))
        elif dataset_type == 'mentions':
            db_record.update(self._transform_mentions_fields(record))
        elif dataset_type == 'gkg':
            db_record.update(self._transform_gkg_fields(record))
        
        return db_record
    
    def _transform_events_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Events dataset specific fields."""
        return {
            # Event identification
            'event_id': record.get('external_id'),
            'event_nature': self._map_event_code(record.get('event_code')),
            'event_summary': self._generate_event_summary(record),
            
            # Geographic information
            'location_name': record.get('location_name'),
            'latitude': record.get('latitude'),
            'longitude': record.get('longitude'),
            'country': record.get('country'),
            
            # Actors
            'primary_actors': [record.get('actor1_name')] if record.get('actor1_name') else [],
            'secondary_actors': [record.get('actor2_name')] if record.get('actor2_name') else [],
            
            # Metrics
            'goldstein_scale': record.get('goldstein_scale'),
            'num_mentions': record.get('num_mentions'),
            'num_sources': record.get('num_sources'),
            'avg_tone': record.get('avg_tone'),
        }
    
    def _transform_mentions_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Mentions dataset specific fields."""
        return {
            'event_id': record.get('external_id'),
            'source_name': record.get('mention_source_name'),
            'mention_type': record.get('mention_type'),
            'confidence_score': record.get('confidence', 0) / 100.0 if record.get('confidence') else None,
            'avg_tone': record.get('mention_doc_tone'),
        }
    
    def _transform_gkg_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transform GKG dataset specific fields."""
        return {
            'event_id': record.get('gkg_record_id'),
            'themes': self._parse_gkg_themes(record.get('themes', '')),
            'locations': self._parse_gkg_locations(record.get('locations', '')),
            'persons': self._parse_gkg_entities(record.get('persons', '')),
            'organizations': self._parse_gkg_entities(record.get('organizations', '')),
            'quotations': self._parse_gkg_quotations(record.get('quotations', '')),
        }
    
    def _get_data_source_id(self) -> str:
        """Get or generate data source ID for GDELT."""
        # This should match the UUID from the data_sources table
        return "gdelt-data-source-uuid"  # Placeholder - should be actual UUID
    
    def _generate_title(self, record: Dict[str, Any], dataset_type: str) -> str:
        """Generate a meaningful title for the record."""
        if dataset_type == 'events':
            event_code = record.get('event_code', 'Unknown')
            location = record.get('location_name', 'Unknown Location')
            return f"GDELT Event {event_code} in {location}"
        elif dataset_type == 'mentions':
            source = record.get('mention_source_name', 'Unknown Source')
            return f"GDELT Mention from {source}"
        elif dataset_type == 'gkg':
            return f"GDELT GKG Record {record.get('gkg_record_id', 'Unknown')}"
        else:
            return f"GDELT {dataset_type.title()} Record"
    
    def _generate_content(self, record: Dict[str, Any], dataset_type: str) -> str:
        """Generate content text from record."""
        content_parts = []
        
        if dataset_type == 'events':
            if record.get('actor1_name'):
                content_parts.append(f"Actor 1: {record['actor1_name']}")
            if record.get('actor2_name'):
                content_parts.append(f"Actor 2: {record['actor2_name']}")
            if record.get('event_code'):
                content_parts.append(f"Event Code: {record['event_code']}")
            if record.get('location_name'):
                content_parts.append(f"Location: {record['location_name']}")
        
        elif dataset_type == 'gkg':
            if record.get('themes'):
                content_parts.append(f"Themes: {record['themes']}")
            if record.get('persons'):
                content_parts.append(f"Persons: {record['persons']}")
            if record.get('organizations'):
                content_parts.append(f"Organizations: {record['organizations']}")
            if record.get('quotations'):
                content_parts.append(f"Quotations: {record['quotations']}")
        
        return "; ".join(content_parts)
    
    def _generate_content_hash(self, record: Dict[str, Any]) -> str:
        """Generate a content hash for deduplication."""
        # Create hash from key fields
        hash_content = json.dumps(record, sort_keys=True, default=str)
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _generate_summary(self, record: Dict[str, Any], dataset_type: str) -> str:
        """Generate a summary of the record."""
        if dataset_type == 'events':
            actor1 = record.get('actor1_name', 'Unknown actor')
            actor2 = record.get('actor2_name', 'another actor')
            location = record.get('location_name', 'an unknown location')
            event_code = record.get('event_code', 'unknown event')
            
            return f"{actor1} involved in event {event_code} with {actor2} in {location}"
        
        return f"GDELT {dataset_type} record"
    
    def _extract_keywords(self, record: Dict[str, Any], dataset_type: str) -> List[str]:
        """Extract keywords from the record."""
        keywords = []
        
        if dataset_type == 'events':
            if record.get('event_code'):
                keywords.append(f"event_{record['event_code']}")
            if record.get('country'):
                keywords.append(record['country'])
            if record.get('actor1_country_code'):
                keywords.append(record['actor1_country_code'])
            if record.get('actor2_country_code'):
                keywords.append(record['actor2_country_code'])
        
        elif dataset_type == 'gkg':
            themes = self._parse_gkg_themes(record.get('themes', ''))
            keywords.extend(themes)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_tags(self, record: Dict[str, Any], dataset_type: str) -> List[str]:
        """Extract tags from the record."""
        tags = [f"gdelt_{dataset_type}"]
        
        if dataset_type == 'events':
            quad_class = record.get('quad_class')
            if quad_class:
                tags.append(f"quad_class_{quad_class}")
        
        return tags
    
    def _calculate_sentiment_score(self, record: Dict[str, Any]) -> Optional[float]:
        """Calculate sentiment score from tone."""
        tone = record.get('avg_tone') or record.get('mention_doc_tone')
        if tone is not None:
            # GDELT tone ranges from -100 to +100, normalize to -1 to +1
            return max(-1.0, min(1.0, tone / 100.0))
        return None
    
    def _build_metadata(self, record: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
        """Build metadata dictionary."""
        metadata = {
            'dataset_type': dataset_type,
            'processing_timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        # Add dataset-specific metadata
        if dataset_type == 'events':
            metadata.update({
                'quad_class': record.get('quad_class'),
                'goldstein_scale': record.get('goldstein_scale'),
                'num_mentions': record.get('num_mentions'),
                'num_sources': record.get('num_sources'),
                'num_articles': record.get('num_articles'),
            })
        
        return metadata
    
    def _map_event_code(self, event_code: str) -> Optional[str]:
        """Map GDELT event code to human-readable description."""
        if not event_code:
            return None
        
        # GDELT CAMEO event codes mapping (simplified)
        event_map = {
            '01': 'Make Public Statement',
            '02': 'Appeal',
            '03': 'Express Intent to Cooperate',
            '04': 'Consult',
            '05': 'Engage in Diplomatic Cooperation',
            '06': 'Engage in Material Cooperation',
            '07': 'Provide Aid',
            '08': 'Yield',
            '09': 'Investigate',
            '10': 'Demand',
            '11': 'Disapprove',
            '12': 'Reject',
            '13': 'Threaten',
            '14': 'Protest',
            '15': 'Exhibit Force Posture',
            '16': 'Reduce Relations',
            '17': 'Coerce',
            '18': 'Assault',
            '19': 'Fight',
            '20': 'Use Unconventional Mass Violence',
        }
        
        # Get first two digits for base category
        base_code = event_code[:2] if len(event_code) >= 2 else event_code
        return event_map.get(base_code, f"Event Code {event_code}")
    
    def _generate_event_summary(self, record: Dict[str, Any]) -> str:
        """Generate detailed event summary."""
        parts = []
        
        if record.get('actor1_name') and record.get('actor2_name'):
            parts.append(f"{record['actor1_name']} and {record['actor2_name']}")
        elif record.get('actor1_name'):
            parts.append(record['actor1_name'])
        
        event_nature = self._map_event_code(record.get('event_code'))
        if event_nature:
            parts.append(f"engaged in: {event_nature}")
        
        if record.get('location_name'):
            parts.append(f"in {record['location_name']}")
        
        return " ".join(parts) if parts else "GDELT event"
    
    def _parse_gkg_themes(self, themes_str: str) -> List[str]:
        """Parse GKG themes string."""
        if not themes_str:
            return []
        
        themes = []
        for theme in themes_str.split(';'):
            if theme.strip():
                # Extract theme name (before any numeric indicators)
                theme_name = re.sub(r',\d+.*$', '', theme).strip()
                if theme_name:
                    themes.append(theme_name)
        
        return themes
    
    def _parse_gkg_locations(self, locations_str: str) -> List[Dict[str, Any]]:
        """Parse GKG locations string."""
        if not locations_str:
            return []
        
        locations = []
        for location in locations_str.split(';'):
            if location.strip():
                parts = location.split(',')
                if len(parts) >= 3:
                    locations.append({
                        'name': parts[0],
                        'country_code': parts[1],
                        'coordinates': parts[2:4] if len(parts) >= 4 else None
                    })
        
        return locations
    
    def _parse_gkg_entities(self, entities_str: str) -> List[str]:
        """Parse GKG entities string (persons/organizations)."""
        if not entities_str:
            return []
        
        entities = []
        for entity in entities_str.split(';'):
            if entity.strip():
                # Extract entity name (before any numeric indicators)
                entity_name = re.sub(r',\d+.*$', '', entity).strip()
                if entity_name:
                    entities.append(entity_name)
        
        return entities
    
    def _parse_gkg_quotations(self, quotations_str: str) -> List[str]:
        """Parse GKG quotations string."""
        if not quotations_str:
            return []
        
        # Split by quotation markers and clean up
        quotations = []
        for quote in quotations_str.split('#'):
            quote = quote.strip()
            if quote and len(quote) > 10:  # Filter out very short quotes
                quotations.append(quote)
        
        return quotations


# Utility functions
async def process_gdelt_file_simple(
    file_path: Path,
    dataset_type: str
) -> Tuple[List[Dict[str, Any]], ProcessingStats]:
    """
    Simple function to process a GDELT file completely.
    
    Args:
        file_path: Path to GDELT file
        dataset_type: Type of dataset ('events', 'mentions', 'gkg')
        
    Returns:
        Tuple of (all_records, processing_stats)
    """
    processor = GDELTFileProcessor()
    return await processor.process_file_complete(file_path, dataset_type)


if __name__ == "__main__":
    # Example usage
    async def main():
        processor = GDELTFileProcessor()
        
        # Example file processing
        file_path = Path("example_gdelt_events.csv")
        if file_path.exists():
            records, stats = await processor.process_file_complete(file_path, "events")
            
            print(f"Processed {stats.total_records} records")
            print(f"Valid: {stats.valid_records}, Invalid: {stats.invalid_records}")
            print(f"Success rate: {stats.success_rate:.1f}%")
            print(f"Processing time: {stats.processing_time_seconds:.2f}s")
            
            if records:
                print(f"Sample record: {records[0]}")
    
    asyncio.run(main())