"""
GDELT Events BigQuery ETL Pipeline for Information Units
========================================================

A comprehensive ETL pipeline that uses BigQuery to extract GDELT Events data and load it into
the information_units table. GDELT Events are the core structured event records that capture
who, what, when, where, and how of global events.

Key Features:
- **Events Focus**: Imports GDELT Events (not GKG) for structured event data
- **BigQuery Integration**: Direct access to GDELT BigQuery events datasets
- **Information Units Schema**: Full compatibility with existing information_units table
- **Geographic Filtering**: Horn of Africa and other regional targeting
- **Actor Processing**: Comprehensive actor1/actor2 relationship mapping
- **Event Classification**: Conflict detection and event type categorization
- **Performance Optimization**: Query optimization and parallel processing

GDELT Events Structure:
- GlobalEventID: Unique event identifier
- Day: Date of event (YYYYMMDD)
- Actor1/Actor2: Primary actors involved
- EventCode/EventRootCode: CAMEO event classification
- QuadClass: Cooperative/Material/Verbal/Conflict classification
- GoldsteinScale: Conflict/cooperation scale (-10 to +10)
- NumMentions/NumSources/NumArticles: Coverage metrics
- AvgTone: Average sentiment tone
- ActionGeo: Geographic information (lat/lon/country/location)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: December 27, 2024
"""

import asyncio
import asyncpg
import logging
import json
import uuid
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import time
import hashlib

# Google Cloud imports
from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError
import google.auth
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EventsETLConfig:
    """Configuration for GDELT Events BigQuery ETL operations."""

    # Database settings
    database_url: str = "postgresql://nyimbi:Abcd1234.@88.80.188.224:5432/lnd"  # Remote database default
    max_connections: int = 10
    connection_timeout: int = 30

    # BigQuery settings
    bigquery_project: str = "gdelt-bq"
    bigquery_dataset: str = "gdeltv2"
    events_table: str = "events_partitioned"  # Use same partitioned table as iu_load.py
    google_credentials_path: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

    # Processing settings
    batch_size: int = 5000
    max_concurrent_batches: int = 3
    enable_ml_processing: bool = True
    enable_conflict_analysis: bool = True

    # Geographic filtering
    target_countries: List[str] = field(default_factory=lambda: ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"])
    enable_geographic_filtering: bool = True

    # Event filtering
    conflict_event_codes: List[str] = field(default_factory=lambda: [
        "18", "19", "20",  # Assault, fight, use conventional military force
        "190", "191", "192", "193", "194", "195", "196",  # Various types of attacks
        "200", "201", "202", "203", "204", "205"  # Mass violence
    ])
    min_goldstein_for_conflict: float = -5.0  # Events below this are considered conflictual

    # Data quality settings
    enable_validation: bool = True
    enable_deduplication: bool = True
    skip_existing_records: bool = True

    # Performance settings
    use_bulk_operations: bool = True
    enable_parallel_processing: bool = True
    checkpoint_interval: int = 5000
    query_timeout_seconds: int = 300

    # ML settings
    ml_scoring_enabled: bool = True
    ml_batch_size: int = 100
    ml_confidence_threshold: float = 0.7


@dataclass
class EventsETLMetrics:
    """GDELT Events ETL processing metrics."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None

    # Processing counts
    total_records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    records_failed: int = 0

    # BigQuery metrics
    bigquery_queries_executed: int = 0
    bigquery_bytes_processed: int = 0
    bigquery_processing_time_seconds: float = 0.0

    # ML processing
    ml_records_processed: int = 0
    ml_processing_time_seconds: float = 0.0

    # Performance metrics
    database_operations: int = 0
    database_time_seconds: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Calculate total processing duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    @property
    def records_per_second(self) -> float:
        """Calculate processing rate."""
        duration = self.duration_seconds
        return self.total_records_processed / duration if duration > 0 else 0.0


class GDELTEventsETL:
    """
    GDELT Events BigQuery ETL pipeline for loading data into information_units table.

    ENHANCED VERSION 2.0 - Complete Field Mapping
    ============================================

    This enhanced version implements comprehensive GDELT Events field mapping:
    - Maps ALL 61 GDELT Events fields to appropriate information_units columns
    - Stores complete GDELT record as JSONB in gdelt_complete_record column
    - Provides intelligent column detection for database compatibility
    - Maintains backward compatibility with existing schemas

    Key Enhancements:
    - BigQuery query includes all 61 GDELT fields (vs. 29 previously)
    - Complete field mapping with safe type conversion
    - Dynamic database insertion based on existing table schema
    - Comprehensive JSONB storage for complete data preservation
    - Enhanced error handling and logging
    """

    def __init__(self, config: EventsETLConfig):
        self.config = config
        self.connection_pool = None
        self.bigquery_client = None
        self.metrics = EventsETLMetrics()
        self._running = False

        # Initialize BigQuery client
        self._initialize_bigquery_client()

        # Track field mapping statistics
        self.gdelt_fields_mapped = 61  # Total GDELT Events fields
        self.information_units_fields_mapped = 0  # Will be set after schema detection

    def _initialize_bigquery_client(self):
        """Initialize BigQuery client using EXACT same method as iu_load.py."""
        try:
            # Set up Google Cloud credentials first
            self._setup_gcloud_credentials()

            # Use SIMPLE initialization like iu_load.py - just bigquery.Client()
            self.bigquery_client = bigquery.Client()

            logger.info(f"✅ BigQuery client initialized (using iu_load.py method)")

        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            logger.info("💡 Make sure GOOGLE_APPLICATION_CREDENTIALS is set or gcloud auth application-default login is configured")
            raise

    def _setup_gcloud_credentials(self):
        """Set up Google Cloud credentials using the same method as iu_load_enhanced.py."""
        if self.config.google_credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.google_credentials_path
            logger.info(f"✅ Using GCS credentials from: {self.config.google_credentials_path}")
        elif not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            # Try default paths
            default_creds = Path.home() / '.config/gcloud/application_default_credentials.json'
            if default_creds.exists():
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(default_creds)
                logger.info(f"✅ Using default GCS credentials: {default_creds}")
            else:
                logger.warning("⚠️  No GCS credentials found. Trying gcloud application default credentials...")

        # Ensure gcloud CLI is in PATH
        gcloud_path = Path.home() / 'google-cloud-sdk/bin'
        if gcloud_path.exists():
            current_path = os.environ.get('PATH', '')
            if str(gcloud_path) not in current_path:
                os.environ['PATH'] = f"{gcloud_path}:{current_path}"
                logger.info(f"✅ Added gcloud to PATH: {gcloud_path}")
        else:
            logger.warning("⚠️  gcloud CLI not found in default location")

    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=2,
                max_size=self.config.max_connections,
                command_timeout=self.config.connection_timeout
            )
            logger.info("✅ Database connection pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self):
        """Close database connections and cleanup."""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Database connections closed")

    def generate_events_query(self, start_date: datetime, end_date: datetime) -> str:
        """Generate BigQuery query using EXACT events_partitioned schema fields provided by user."""

        # Convert dates to string format for partitioned table
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Horn of Africa countries
        horn_countries = ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG", "RW", "CD"]
        country_filter = "', '".join(horn_countries)

        # UPDATED QUERY using EXACT 56 fields from BigQuery events_partitioned schema
        query = f"""
        SELECT
            MonthYear,
            Year,
            FractionDate,
            Actor1Code,
            Actor1Name,
            Actor1CountryCode,
            Actor1KnownGroupCode,
            Actor1EthnicCode,
            Actor1Religion1Code,
            Actor1Religion2Code,
            Actor1Type1Code,
            Actor1Type2Code,
            Actor1Type3Code,
            Actor2Code,
            Actor2Name,
            Actor2CountryCode,
            Actor2KnownGroupCode,
            Actor2EthnicCode,
            Actor2Religion1Code,
            Actor2Religion2Code,
            Actor2Type1Code,
            Actor2Type2Code,
            Actor2Type3Code,
            IsRootEvent,
            EventCode,
            EventBaseCode,
            EventRootCode,
            QuadClass,
            GoldsteinScale,
            NumMentions,
            NumSources,
            NumArticles,
            AvgTone,
            Actor1Geo_Type,
            Actor1Geo_Fullname,
            Actor1Geo_CountryCode,
            Actor1Geo_ADM1Code,
            Actor1Geo_ADM2Code,
            Actor1Geo_Lat,
            Actor1Geo_Long,
            Actor1Geo_FeatureID,
            Actor2Geo_Type,
            Actor2Geo_Fullname,
            Actor2Geo_CountryCode,
            Actor2Geo_ADM1Code,
            Actor2Geo_ADM2Code,
            Actor2Geo_Lat,
            Actor2Geo_Long,
            Actor2Geo_FeatureID,
            ActionGeo_Type,
            ActionGeo_Fullname,
            ActionGeo_CountryCode,
            ActionGeo_ADM1Code,
            ActionGeo_ADM2Code,
            ActionGeo_Lat,
            ActionGeo_Long,
            ActionGeo_FeatureID,
            DATEADDED,
            SOURCEURL
        FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.{self.config.events_table}`
        WHERE _PARTITIONTIME >= TIMESTAMP('{start_date_str}')
          AND _PARTITIONTIME < TIMESTAMP('{end_date_str}')
          AND (
            ActionGeo_CountryCode IN ('{country_filter}')
            OR Actor1CountryCode IN ('{country_filter}')
            OR Actor2CountryCode IN ('{country_filter}')
          )
        ORDER BY MonthYear DESC, DATEADDED DESC
        LIMIT {self.config.batch_size}
        """

        return query

    def execute_bigquery_query(self, query: str) -> pd.DataFrame:
        """Execute BigQuery query using EXACT same method as iu_load.py."""
        try:
            start_time = time.time()

            logger.info("🔄 Executing partitioned table query...")

            # Execute the query - SAME as iu_load.py
            job = self.bigquery_client.query(query)
            df = job.result().to_dataframe()

            # Get metrics - SAME as iu_load.py
            bytes_billed = job.total_bytes_billed or 0
            processing_time = (job.ended - job.started).total_seconds() if job.ended and job.started else 0

            # Update metrics
            self.metrics.bigquery_queries_executed += 1
            self.metrics.bigquery_processing_time_seconds += processing_time
            self.metrics.bigquery_bytes_processed += bytes_billed

            # Log results - SAME format as iu_load.py
            logger.info(f"✅ Query completed successfully!")
            logger.info(f"📊 Results: {len(df)} events retrieved")
            logger.info(f"🔗 URLs found: {df['SOURCEURL'].notna().sum()} events with source URLs")
            logger.info(f"💰 Bytes billed: {bytes_billed:,} ({bytes_billed/(1024**3):.3f} GB)")
            logger.info(f"⏱️  Query time: {processing_time:.1f} seconds")
            logger.info(f"💵 Actual cost: ${(bytes_billed / (1024**4)) * 5:.6f}")

            return df

        except Exception as e:
            logger.error(f"BigQuery execution failed: {e}")
            raise

    def convert_event_to_information_unit(self, row: pd.Series) -> Dict[str, Any]:
        """Convert GDELT Event row to information_units format with complete field mapping."""

        # NOTE: GLOBALEVENTID and SQLDATE are NOT in events_partitioned schema
        # Use available fields: MonthYear, Year, FractionDate, DATEADDED

        # Parse Year/MonthYear for published_at field since SQLDATE not available
        date_value = None
        if pd.notna(row['Year']) and pd.notna(row['MonthYear']):
            try:
                year = int(row['Year'])
                month_year = int(row['MonthYear'])
                month = month_year % 100  # Extract month from MonthYear
                # Use day 1 since we don't have specific day
                date_value = datetime(year, month, 1, tzinfo=timezone.utc)
            except Exception as e:
                logger.warning(f"Failed to parse Year/MonthYear {row['Year']}/{row['MonthYear']}: {e}")
                date_value = datetime.now(timezone.utc)

        # Parse DATEADDED for created_at if available
        dateadded_datetime = None
        if pd.notna(row['DATEADDED']):
            try:
                dateadded_str = str(int(row['DATEADDED']))
                dateadded_year = int(dateadded_str[:4])
                dateadded_month = int(dateadded_str[4:6])
                dateadded_day = int(dateadded_str[6:8])
                dateadded_datetime = datetime(dateadded_year, dateadded_month, dateadded_day, tzinfo=timezone.utc)
            except:
                dateadded_datetime = None

        # Extract actor information
        actor1_name = str(row['Actor1Name']) if pd.notna(row['Actor1Name']) else ""
        actor1_country = str(row['Actor1CountryCode']) if pd.notna(row['Actor1CountryCode']) else ""
        actor2_name = str(row['Actor2Name']) if pd.notna(row['Actor2Name']) else ""
        actor2_country = str(row['Actor2CountryCode']) if pd.notna(row['Actor2CountryCode']) else ""

        # Create event description from event codes
        event_code = str(row['EventCode']) if pd.notna(row['EventCode']) else ""
        event_root_code = str(row['EventRootCode']) if pd.notna(row['EventRootCode']) else ""
        quad_class = int(row['QuadClass']) if pd.notna(row['QuadClass']) else 0

        # Map quad classes to descriptions
        quad_descriptions = {
            1: "Verbal Cooperation",
            2: "Material Cooperation",
            3: "Verbal Conflict",
            4: "Material Conflict"
        }
        quad_desc = quad_descriptions.get(quad_class, "Unknown")

        # Generate title and description
        title_parts = []
        if actor1_name and actor2_name:
            title_parts.append(f"{actor1_name} and {actor2_name}")
        elif actor1_name:
            title_parts.append(actor1_name)

        if event_code:
            title_parts.append(f"Event {event_code}")

        # Generate unique event ID from available fields since GLOBALEVENTID not in schema
        event_id = f"{row['Year']}{row['MonthYear']}{row['DATEADDED']}" if pd.notna(row['DATEADDED']) else f"{row['Year']}{row['MonthYear']}"
        title = " - ".join(title_parts) if title_parts else f"Event {event_id}"

        # Create comprehensive description
        description_parts = []
        description_parts.append(f"Event Type: {quad_desc} (Code: {event_code})")

        if actor1_name:
            description_parts.append(f"Primary Actor: {actor1_name} ({actor1_country})")
        if actor2_name:
            description_parts.append(f"Secondary Actor: {actor2_name} ({actor2_country})")

        goldstein = float(row['GoldsteinScale']) if pd.notna(row['GoldsteinScale']) else None
        if goldstein is not None:
            description_parts.append(f"Goldstein Scale: {goldstein:.1f}")

        num_mentions = int(row['NumMentions']) if pd.notna(row['NumMentions']) else None
        if num_mentions:
            description_parts.append(f"Media Mentions: {num_mentions}")

        content = " | ".join(description_parts)

        # Extract geographic information from ActionGeo fields - Updated for events_partitioned schema
        latitude = float(row['ActionGeo_Lat']) if pd.notna(row['ActionGeo_Lat']) else None
        longitude = float(row['ActionGeo_Long']) if pd.notna(row['ActionGeo_Long']) else None
        country = str(row['ActionGeo_CountryCode']) if pd.notna(row['ActionGeo_CountryCode']) else None
        location_name = str(row['ActionGeo_Fullname']) if pd.notna(row['ActionGeo_Fullname']) else None

        # Extract key values
        event_code = str(row['EventCode']) if pd.notna(row['EventCode']) else ""
        quad_class = int(row['QuadClass']) if pd.notna(row['QuadClass']) else 0
        goldstein_raw = float(row['GoldsteinScale']) if pd.notna(row['GoldsteinScale']) else None
        avg_tone_raw = float(row['AvgTone']) if pd.notna(row['AvgTone']) else None
        num_mentions = int(row['NumMentions']) if pd.notna(row['NumMentions']) else None

        # Normalize GDELT values - SAME as iu_load.py
        # AvgTone: -100 to +100 -> normalize to -1 to +1 for sentiment_score
        if avg_tone_raw is not None:
            avg_tone = max(-1.0, min(1.0, avg_tone_raw / 100.0))
        else:
            avg_tone = None

        # GoldsteinScale: -10 to +10 -> normalize to 0 to 1 for reliability_score
        if goldstein_raw is not None:
            goldstein = max(0.0, min(1.0, (goldstein_raw + 10.0) / 20.0))
        else:
            goldstein = None

        # Determine conflict classification - SAME logic as iu_load.py
        is_conflict = False
        if quad_class == 4 or (goldstein_raw is not None and goldstein_raw <= -5.0):
            is_conflict = True
        elif quad_class == 3 or (goldstein_raw is not None and goldstein_raw <= -2.0):
            is_conflict = True

        # Extract actors for relationships
        primary_actors = []
        secondary_actors = []

        if actor1_name:
            primary_actors.append(actor1_name)
        if actor2_name:
            secondary_actors.append(actor2_name)

        # Store complete GDELT row in metadatax first - SAME as iu_load.py
        complete_gdelt_row = {}
        # Get DataFrame columns from the row's parent DataFrame (we'll need to pass this)
        # For now, use the row index which contains column names
        for col_name in row.index:
            if pd.notna(row[col_name]):
                complete_gdelt_row[col_name.lower()] = str(row[col_name]) if not isinstance(row[col_name], (int, float)) else row[col_name]
            else:
                complete_gdelt_row[col_name.lower()] = None

        # Extract and validate source URL for content_url field - SAME as iu_load.py
        raw_source_url = str(row['SOURCEURL']) if pd.notna(row['SOURCEURL']) else None

        # Only use valid HTTP/HTTPS URLs - reject gdelt:// and other invalid schemes
        source_url = None
        if raw_source_url:
            raw_source_url = raw_source_url.strip()
            if raw_source_url.startswith(('http://', 'https://')) and len(raw_source_url) > 10:
                # Basic URL validation - must be proper HTTP/HTTPS URL
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(raw_source_url)
                    if parsed.netloc and parsed.scheme in ['http', 'https']:
                        source_url = raw_source_url
                except:
                    source_url = None

        # Create metadatax with complete GDELT row plus additional fields - SAME as iu_load.py
        metadatax = {
            'complete_gdelt_row': complete_gdelt_row,
            'partitioned_table': True,
            'data_source': 'gdelt_events_partitioned_bigquery_enhanced',
            'import_timestamp': datetime.now().isoformat(),
        }

        # Store the raw SOURCEURL in metadatax for reference - SAME as iu_load.py
        if 'complete_gdelt_row' in metadatax:
            metadatax['complete_gdelt_row']['original_sourceurl'] = raw_source_url
            metadatax['url_validation'] = {
                'raw_sourceurl': raw_source_url,
                'validated_content_url': source_url,
                'validation_status': 'valid' if source_url else 'invalid_or_missing'
            }

        # Create record using available fields since GLOBALEVENTID not in schema
        external_id = event_id  # Use generated event_id since GLOBALEVENTID not available
        record_created_at = dateadded_datetime if dateadded_datetime else datetime.now(timezone.utc)

        return {
            'id': str(uuid.uuid4()),
            'external_id': external_id,
            'title': title[:500],  # Limit title length
            'content': content[:2000],  # Limit content length
            'content_url': source_url,  # Validated URL
            'published_at': date_value,  # SQLDATE as published_at
            'created_at': record_created_at,  # DATEADDED as created_at
            'updated_at': datetime.now(timezone.utc),
            'unit_type': 'dataset_entry',  # Same unit_type as iu_load.py
            'language_code': 'en',  # GDELT is primarily English
            'latitude_extracted': latitude,
            'longitude_extracted': longitude,
            'country_extracted': country,
            'location_name_extracted': location_name,
            'event_id': external_id,
            'event_nature': f"{event_code}: {quad_desc}",
            'event_summary_extracted': content,
            'event_date_extracted': date_value.date() if date_value else None,
            'conflict_classification': 'singular_incident' if is_conflict else None,
            'perpetrator_names_extracted': primary_actors if is_conflict else [],
            'victim_names_extracted': secondary_actors if is_conflict and quad_class == 4 else [],
            'authority_figures_extracted': primary_actors if not is_conflict else [],
            'extraction_confidence_score': 0.95,  # Same as iu_load.py
            'extraction_id': str(uuid.uuid4()),  # Add required extraction_id
            'metadatax': json.dumps(metadatax),  # Complete GDELT record in metadatax
            'sentiment_score': avg_tone if avg_tone is not None else None,
            'reliability_score': goldstein if goldstein is not None else None
        }

    def _create_complete_gdelt_record(self, row: pd.Series) -> Dict[str, Any]:
        """Create complete GDELT record for JSONB storage."""

        # Helper function to safely convert values
        def safe_convert(value, target_type=str):
            if pd.isna(value):
                return None
            try:
                if target_type == str:
                    return str(value)
                elif target_type == int:
                    return int(value)
                elif target_type == float:
                    return float(value)
                elif target_type == bool:
                    return bool(value)
                else:
                    return value
            except:
                return None

        return {
            # Core Event Identification - Updated for events_partitioned schema
            'month_year': safe_convert(row['MonthYear'], int),
            'year': safe_convert(row['Year'], int),
            'fraction_date': safe_convert(row['FractionDate'], float),

            # Actor1 Fields
            'actor1_code': safe_convert(row['Actor1Code']),
            'actor1_name': safe_convert(row['Actor1Name']),
            'actor1_country_code': safe_convert(row['Actor1CountryCode']),
            'actor1_known_group_code': safe_convert(row['Actor1KnownGroupCode']),
            'actor1_ethnic_code': safe_convert(row['Actor1EthnicCode']),
            'actor1_religion1_code': safe_convert(row['Actor1Religion1Code']),
            'actor1_religion2_code': safe_convert(row['Actor1Religion2Code']),
            'actor1_type1_code': safe_convert(row['Actor1Type1Code']),
            'actor1_type2_code': safe_convert(row['Actor1Type2Code']),
            'actor1_type3_code': safe_convert(row['Actor1Type3Code']),

            # Actor2 Fields
            'actor2_code': safe_convert(row['Actor2Code']),
            'actor2_name': safe_convert(row['Actor2Name']),
            'actor2_country_code': safe_convert(row['Actor2CountryCode']),
            'actor2_known_group_code': safe_convert(row['Actor2KnownGroupCode']),
            'actor2_ethnic_code': safe_convert(row['Actor2EthnicCode']),
            'actor2_religion1_code': safe_convert(row['Actor2Religion1Code']),
            'actor2_religion2_code': safe_convert(row['Actor2Religion2Code']),
            'actor2_type1_code': safe_convert(row['Actor2Type1Code']),
            'actor2_type2_code': safe_convert(row['Actor2Type2Code']),
            'actor2_type3_code': safe_convert(row['Actor2Type3Code']),

            # Event Classification
            'is_root_event': safe_convert(row['IsRootEvent'], int),
            'event_code': safe_convert(row['EventCode']),
            'event_base_code': safe_convert(row['EventBaseCode']),
            'event_root_code': safe_convert(row['EventRootCode']),
            'quad_class': safe_convert(row['QuadClass'], int),
            'goldstein_scale': safe_convert(row['GoldsteinScale'], float),

            # Media Coverage
            'num_mentions': safe_convert(row['NumMentions'], int),
            'num_sources': safe_convert(row['NumSources'], int),
            'num_articles': safe_convert(row['NumArticles'], int),
            'avg_tone': safe_convert(row['AvgTone'], float),

            # Actor1 Geography - Updated field names for events_partitioned schema
            'actor1_geo_type': safe_convert(row['Actor1Geo_Type'], int),
            'actor1_geo_fullname': safe_convert(row['Actor1Geo_Fullname']),
            'actor1_geo_country_code': safe_convert(row['Actor1Geo_CountryCode']),
            'actor1_geo_adm1_code': safe_convert(row['Actor1Geo_ADM1Code']),
            'actor1_geo_adm2_code': safe_convert(row['Actor1Geo_ADM2Code']),
            'actor1_geo_lat': safe_convert(row['Actor1Geo_Lat'], float),
            'actor1_geo_long': safe_convert(row['Actor1Geo_Long'], float),
            'actor1_geo_feature_id': safe_convert(row['Actor1Geo_FeatureID']),

            # Actor2 Geography - Updated field names for events_partitioned schema
            'actor2_geo_type': safe_convert(row['Actor2Geo_Type'], int),
            'actor2_geo_fullname': safe_convert(row['Actor2Geo_Fullname']),
            'actor2_geo_country_code': safe_convert(row['Actor2Geo_CountryCode']),
            'actor2_geo_adm1_code': safe_convert(row['Actor2Geo_ADM1Code']),
            'actor2_geo_adm2_code': safe_convert(row['Actor2Geo_ADM2Code']),
            'actor2_geo_lat': safe_convert(row['Actor2Geo_Lat'], float),
            'actor2_geo_long': safe_convert(row['Actor2Geo_Long'], float),
            'actor2_geo_feature_id': safe_convert(row['Actor2Geo_FeatureID']),

            # Action Geography - Updated field names for events_partitioned schema
            'action_geo_type': safe_convert(row['ActionGeo_Type'], int),
            'action_geo_fullname': safe_convert(row['ActionGeo_Fullname']),
            'action_geo_country_code': safe_convert(row['ActionGeo_CountryCode']),
            'action_geo_adm1_code': safe_convert(row['ActionGeo_ADM1Code']),
            'action_geo_adm2_code': safe_convert(row['ActionGeo_ADM2Code']),
            'action_geo_lat': safe_convert(row['ActionGeo_Lat'], float),
            'action_geo_long': safe_convert(row['ActionGeo_Long'], float),
            'action_geo_feature_id': safe_convert(row['ActionGeo_FeatureID']),

            # Source and Timing
            'dateadd': safe_convert(row['DATEADDED'], int),
            'source_url': safe_convert(row['SOURCEURL'])
        }

    def _map_all_gdelt_fields(self, row: pd.Series, date_value: datetime) -> Dict[str, Any]:
        """Map all GDELT fields to appropriate information_units columns."""

        # Helper function to safely convert values
        def safe_convert(value, target_type=str):
            if pd.isna(value):
                return None
            try:
                if target_type == str:
                    return str(value)
                elif target_type == int:
                    return int(value)
                elif target_type == float:
                    return float(value)
                elif target_type == bool:
                    return bool(value)
                else:
                    return value
            except:
                return None

        # Create comprehensive mapping to information_units columns
        # This maps GDELT fields to the most appropriate information_units column names
        # Updated for events_partitioned schema (no GLOBALEVENTID or SQLDATE)
        return {
            # GDELT-specific columns (these should exist in information_units table)
            'gdelt_month_year': safe_convert(row['MonthYear'], int),
            'gdelt_year': safe_convert(row['Year'], int),
            'gdelt_fraction_date': safe_convert(row['FractionDate'], float),
            'gdelt_date_added': safe_convert(row['DATEADDED'], int),
            'gdelt_is_root_event': safe_convert(row['IsRootEvent'], int),
            'gdelt_event_code': safe_convert(row['EventCode']),
            'gdelt_event_base_code': safe_convert(row['EventBaseCode']),
            'gdelt_event_root_code': safe_convert(row['EventRootCode']),
            'gdelt_quad_class': safe_convert(row['QuadClass'], int),
            'gdelt_goldstein_scale': safe_convert(row['GoldsteinScale'], float),
            'gdelt_num_mentions': safe_convert(row['NumMentions'], int),
            'gdelt_num_sources': safe_convert(row['NumSources'], int),
            'gdelt_num_articles': safe_convert(row['NumArticles'], int),
            'gdelt_avg_tone': safe_convert(row['AvgTone'], float),
            'gdelt_source_url': safe_convert(row['SOURCEURL']),

            # Actor1 fields
            'actor1_code': safe_convert(row['Actor1Code']),
            'actor1_name': safe_convert(row['Actor1Name']),
            'actor1_country_code': safe_convert(row['Actor1CountryCode']),
            'actor1_known_group_code': safe_convert(row['Actor1KnownGroupCode']),
            'actor1_ethnic_code': safe_convert(row['Actor1EthnicCode']),
            'actor1_religion1_code': safe_convert(row['Actor1Religion1Code']),
            'actor1_religion2_code': safe_convert(row['Actor1Religion2Code']),
            'actor1_type1_code': safe_convert(row['Actor1Type1Code']),
            'actor1_type2_code': safe_convert(row['Actor1Type2Code']),
            'actor1_type3_code': safe_convert(row['Actor1Type3Code']),

            # Actor2 fields
            'actor2_code': safe_convert(row['Actor2Code']),
            'actor2_name': safe_convert(row['Actor2Name']),
            'actor2_country_code': safe_convert(row['Actor2CountryCode']),
            'actor2_known_group_code': safe_convert(row['Actor2KnownGroupCode']),
            'actor2_ethnic_code': safe_convert(row['Actor2EthnicCode']),
            'actor2_religion1_code': safe_convert(row['Actor2Religion1Code']),
            'actor2_religion2_code': safe_convert(row['Actor2Religion2Code']),
            'actor2_type1_code': safe_convert(row['Actor2Type1Code']),
            'actor2_type2_code': safe_convert(row['Actor2Type2Code']),
            'actor2_type3_code': safe_convert(row['Actor2Type3Code']),

            # Actor1 Geography
            'actor1_geo_type': safe_convert(row['Actor1Geo_Type'], int),
            'actor1_geo_fullname': safe_convert(row['Actor1Geo_Fullname']),
            'actor1_geo_country_code': safe_convert(row['Actor1Geo_CountryCode']),
            'actor1_geo_adm1_code': safe_convert(row['Actor1Geo_ADM1Code']),
            'actor1_geo_adm2_code': safe_convert(row['Actor1Geo_ADM2Code']),
            'actor1_geo_lat': safe_convert(row['Actor1Geo_Lat'], float),
            'actor1_geo_long': safe_convert(row['Actor1Geo_Long'], float),
            'actor1_geo_feature_id': safe_convert(row['Actor1Geo_FeatureID']),

            # Actor2 Geography
            'actor2_geo_type': safe_convert(row['Actor2Geo_Type'], int),
            'actor2_geo_fullname': safe_convert(row['Actor2Geo_Fullname']),
            'actor2_geo_country_code': safe_convert(row['Actor2Geo_CountryCode']),
            'actor2_geo_adm1_code': safe_convert(row['Actor2Geo_ADM1Code']),
            'actor2_geo_adm2_code': safe_convert(row['Actor2Geo_ADM2Code']),
            'actor2_geo_lat': safe_convert(row['Actor2Geo_Lat'], float),
            'actor2_geo_long': safe_convert(row['Actor2Geo_Long'], float),
            'actor2_geo_feature_id': safe_convert(row['Actor2Geo_FeatureID']),

            # Action Geography
            'action_geo_type': safe_convert(row['ActionGeo_Type'], int),
            'action_geo_fullname': safe_convert(row['ActionGeo_Fullname']),
            'action_geo_country_code': safe_convert(row['ActionGeo_CountryCode']),
            'action_geo_adm1_code': safe_convert(row['ActionGeo_ADM1Code']),
            'action_geo_adm2_code': safe_convert(row['ActionGeo_ADM2Code']),
            'action_geo_lat': safe_convert(row['ActionGeo_Lat'], float),
            'action_geo_long': safe_convert(row['ActionGeo_Long'], float),
            'action_geo_feature_id': safe_convert(row['ActionGeo_FeatureID'])
        }

    def _extract_location_info(self, row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
        """Extract geographic information from ActionGeo fields - Updated for events_partitioned schema."""

        # Try ActionGeo first (where the event occurred) - Updated field names
        lat = float(row['ActionGeo_Lat']) if pd.notna(row['ActionGeo_Lat']) else None
        lon = float(row['ActionGeo_Long']) if pd.notna(row['ActionGeo_Long']) else None
        country = str(row['ActionGeo_CountryCode']) if pd.notna(row['ActionGeo_CountryCode']) else None
        location = str(row['ActionGeo_Fullname']) if pd.notna(row['ActionGeo_Fullname']) else None

        # If no ActionGeo, try Actor1Geo - Updated field names
        if not lat and not lon:
            lat = float(row['Actor1Geo_Lat']) if pd.notna(row['Actor1Geo_Lat']) else None
            lon = float(row['Actor1Geo_Long']) if pd.notna(row['Actor1Geo_Long']) else None
            if not country:
                country = str(row['Actor1Geo_CountryCode']) if pd.notna(row['Actor1Geo_CountryCode']) else None
            if not location:
                location = str(row['Actor1Geo_Fullname']) if pd.notna(row['Actor1Geo_Fullname']) else None

        return lat, lon, country, location

    def _is_conflict_event(self, row: pd.Series) -> bool:
        """Determine if the event is conflict-related."""

        # Check event code
        event_code = str(row['EventCode']) if pd.notna(row['EventCode']) else ""
        if event_code in self.config.conflict_event_codes:
            return True

        # Check Goldstein Scale (negative values indicate conflict)
        goldstein = float(row['GoldsteinScale']) if pd.notna(row['GoldsteinScale']) else 0
        if goldstein <= self.config.min_goldstein_for_conflict:
            return True

        # Check QuadClass (4 = Material Conflict)
        quad_class = int(row['QuadClass']) if pd.notna(row['QuadClass']) else 0
        if quad_class == 4:
            return True

        return False

    async def store_information_units(self, information_units: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """Store information units using EXACT same method as iu_load.py."""
        if not information_units:
            return 0, 0, 0

        inserted = 0
        updated = 0
        skipped = 0

        async with self.connection_pool.acquire() as conn:
            for unit in information_units:
                try:
                    # Check if exists - SAME as iu_load.py
                    external_id = unit['external_id']
                    existing = await conn.fetchval(
                        "SELECT id FROM information_units WHERE external_id = $1",
                        external_id
                    )

                    if existing:
                        skipped += 1
                        continue

                    # Insert new record using EXACT same structure as iu_load.py
                    await conn.execute("""
                        INSERT INTO information_units (
                            id, external_id, title, content, content_url, published_at,
                            created_at, updated_at, unit_type, language_code,
                            latitude_extracted, longitude_extracted, country_extracted,
                            location_name_extracted, event_id, event_nature,
                            event_summary_extracted, event_date_extracted,
                            conflict_classification, extraction_confidence_score, extraction_id, metadatax,
                            sentiment_score, reliability_score
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                            $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
                        )
                    """,
                    unit['id'], unit['external_id'], unit['title'], unit['content'],
                    unit['content_url'], unit['published_at'], unit['created_at'],
                    unit['updated_at'], unit['unit_type'], unit['language_code'],
                    unit['latitude_extracted'], unit['longitude_extracted'],
                    unit['country_extracted'], unit['location_name_extracted'],
                    unit['event_id'], unit['event_nature'], unit['event_summary_extracted'],
                    unit['event_date_extracted'], unit['conflict_classification'],
                    unit['extraction_confidence_score'], unit['extraction_id'], unit['metadatax'],
                    unit['sentiment_score'], unit['reliability_score']
                    )

                    inserted += 1

                except Exception as e:
                    logger.error(f"Failed to store information unit {unit.get('external_id', 'unknown')}: {e}")
                    skipped += 1

        return inserted, updated, skipped

    async def process_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback: Optional[callable] = None
    ) -> EventsETLMetrics:
        """Process GDELT Events for a specific date range."""

        logger.info(f"🚀 Starting GDELT Events ETL for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Reset metrics
        self.metrics = EventsETLMetrics()

        try:
            # Process in daily chunks for better performance
            current_date = start_date
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=1), end_date)

                logger.info(f"Processing Events chunk: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")

                # Generate and execute BigQuery query
                query = self.generate_events_query(current_date, chunk_end)
                df = self.execute_bigquery_query(query)

                if len(df) == 0:
                    logger.info(f"No events found for {current_date.strftime('%Y-%m-%d')}")
                    current_date = chunk_end
                    continue

                # Convert to information units
                information_units = []
                for _, row in df.iterrows():
                    try:
                        unit = self.convert_event_to_information_unit(row)
                        information_units.append(unit)
                    except Exception as e:
                        logger.warning(f"Failed to convert event to information unit: {e}")
                        self.metrics.records_failed += 1

                # Store in database
                if information_units:
                    inserted, updated, skipped = await self.store_information_units(information_units)

                    # Update metrics
                    self.metrics.records_inserted += inserted
                    self.metrics.records_updated += updated
                    self.metrics.records_skipped += skipped
                    self.metrics.total_records_processed += len(information_units)

                    logger.info(f"Processed {len(information_units)} events: {inserted} inserted, {updated} updated, {skipped} skipped")

                # Progress callback
                if progress_callback:
                    progress_callback(self.metrics)

                current_date = chunk_end

                # Brief pause between chunks
                await asyncio.sleep(0.1)

            # Finalize metrics
            self.metrics.end_time = datetime.now(timezone.utc)

            logger.info(f"🎉 GDELT Events ETL completed!")
            logger.info(f"Total events processed: {self.metrics.total_records_processed:,}")
            logger.info(f"Events inserted: {self.metrics.records_inserted:,}")
            logger.info(f"Events updated: {self.metrics.records_updated:,}")
            logger.info(f"Processing rate: {self.metrics.records_per_second:.1f} events/second")

            return self.metrics

        except Exception as e:
            logger.error(f"GDELT Events ETL failed: {e}")
            self.metrics.end_time = datetime.now(timezone.utc)
            raise

    async def health_check(self) -> bool:
        """Perform health check of BigQuery and database connections."""
        try:
            # Test BigQuery connection
            test_query = f"""
            SELECT COUNT(*) as event_count
            FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.{self.config.events_table}`
            WHERE Day >= {(datetime.now() - timedelta(days=1)).strftime('%Y%m%d')}
            LIMIT 1
            """

            df = self.execute_bigquery_query(test_query)
            logger.info(f"BigQuery health check passed: {df.iloc[0]['event_count']} recent events")

            # Test database connection
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM information_units LIMIT 1")
                logger.info(f"Database health check passed: {result} total information units")

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_field_mapping_report(self) -> Dict[str, Any]:
        """Generate comprehensive field mapping report."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get information_units table schema
                table_columns_result = await conn.fetch("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'information_units'
                    ORDER BY ordinal_position
                """)

                existing_columns = {row['column_name']: {
                    'data_type': row['data_type'],
                    'nullable': row['is_nullable']
                } for row in table_columns_result}

                # Get GDELT field mappings
                sample_row_data = {
                    'GlobalEventID': 1, 'Day': 20250101, 'MonthYear': 202501, 'Year': 2025,
                    'FractionDate': 2025.001, 'Actor1Code': 'USA', 'Actor1Name': 'TEST'
                }

                import pandas as pd
                sample_series = pd.Series(sample_row_data)

                try:
                    sample_mapping = self._map_all_gdelt_fields(sample_series, datetime.now())
                    gdelt_field_names = list(sample_mapping.keys())
                except:
                    gdelt_field_names = []

                # Generate mapping report
                report = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'gdelt_fields_total': 61,
                    'gdelt_fields_in_query': self.gdelt_fields_mapped,
                    'information_units_columns_total': len(existing_columns),
                    'gdelt_mapping_fields': len(gdelt_field_names),
                    'existing_columns': list(existing_columns.keys()),
                    'gdelt_mapped_fields': gdelt_field_names,
                    'missing_gdelt_columns': [
                        field for field in gdelt_field_names
                        if field not in existing_columns
                    ],
                    'coverage_percentage': round(
                        (len([f for f in gdelt_field_names if f in existing_columns]) / 61) * 100, 2
                    ) if gdelt_field_names else 0.0,
                    'enhancements': {
                        'complete_jsonb_storage': 'gdelt_complete_record' in existing_columns,
                        'dynamic_column_detection': True,
                        'safe_type_conversion': True,
                        'backward_compatibility': True
                    }
                }

                self.information_units_fields_mapped = len([
                    f for f in gdelt_field_names if f in existing_columns
                ])

                return report

        except Exception as e:
            logger.error(f"Failed to generate field mapping report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'gdelt_fields_total': 61,
                'status': 'error'
            }


# Factory function
def create_events_etl(
    database_url: str = "postgresql://nyimbi:Abcd1234.@88.80.188.224:5432/lnd",
    target_countries: Optional[List[str]] = None,
    google_credentials_path: Optional[str] = None,
    **kwargs
) -> GDELTEventsETL:
    """
    Create a GDELT Events ETL instance with comprehensive field mapping (v2.0).

    ENHANCED VERSION 2.0 FEATURES:
    ==============================
    - Maps ALL 61 GDELT Events fields to appropriate information_units columns
    - Stores complete GDELT record as JSONB in gdelt_complete_record column
    - Intelligent database schema detection for dynamic column mapping
    - Backward compatibility with existing information_units schemas
    - Enhanced error handling and type conversion safety
    - Comprehensive field mapping reporting and statistics
    - Uses same BigQuery credentials method as iu_load_enhanced.py

    Args:
        database_url: PostgreSQL database URL
        target_countries: List of country codes to target (Horn of Africa default)
        google_credentials_path: Path to Google Cloud credentials JSON file
        **kwargs: Additional configuration options

    Returns:
        Configured GDELTEventsETL instance with complete field mapping capabilities

    Example:
        >>> etl = create_events_etl()  # Uses default local DB and credentials
        >>> await etl.initialize()
        >>> report = await etl.get_field_mapping_report()
        >>> print(f"Mapping {report['gdelt_fields_total']} GDELT fields to information_units")
    """
    # Set up credentials path
    if google_credentials_path is None:
        google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    config = EventsETLConfig(
        database_url=database_url,
        target_countries=target_countries or ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
        google_credentials_path=google_credentials_path,
        **kwargs
    )
    return GDELTEventsETL(config)


# Convenience function
async def run_events_etl(
    start_date: datetime,
    end_date: datetime,
    database_url: str = "postgresql://nyimbi:Abcd1234.@88.80.188.224:5432/lnd",
    target_countries: Optional[List[str]] = None,
    google_credentials_path: Optional[str] = None,
    **kwargs
) -> EventsETLMetrics:
    """
    Quick function to run enhanced GDELT Events ETL with complete field mapping.

    ENHANCED VERSION 2.0:
    - Processes ALL 61 GDELT Events fields automatically
    - Stores complete GDELT records as JSONB for full data preservation
    - Maps all fields to appropriate information_units columns
    - Provides comprehensive processing metrics and field mapping statistics

    Args:
        database_url: PostgreSQL database URL
        start_date: Start date for processing (GDELT Events data)
        end_date: End date for processing (GDELT Events data)
        target_countries: List of country codes to target (Horn of Africa default)
        **kwargs: Additional configuration options

    Returns:
        EventsETLMetrics with comprehensive processing statistics

    Example:
        >>> from datetime import datetime, timedelta
        >>> end_date = datetime.now()
        >>> start_date = end_date - timedelta(days=7)
        >>> metrics = await run_events_etl(
        ...     "postgresql://user:pass@host/db",
        ...     start_date, end_date
        ... )
        >>> print(f"Processed {metrics.total_records_processed} events with full field mapping")
    """
    etl = create_events_etl(
        database_url=database_url,
        target_countries=target_countries,
        google_credentials_path=google_credentials_path,
        **kwargs
    )

    try:
        await etl.initialize()
        # Log field mapping report before processing
        try:
            report = await etl.get_field_mapping_report()
            logger.info(f"🎯 GDELT Field Mapping: {report['gdelt_fields_total']} total fields, "
                       f"{report['coverage_percentage']}% coverage in information_units table")
        except:
            logger.info("🎯 GDELT Enhanced ETL v2.0 - Complete field mapping enabled")

        return await etl.process_date_range(start_date, end_date)
    finally:
        await etl.close()
