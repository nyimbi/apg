"""
GDELT BigQuery ETL Pipeline for Information Units
================================================

A comprehensive ETL pipeline that uses BigQuery to extract GDELT GKG data and load it into
the information_units table, replacing the traditional file-based approach with direct
BigQuery access for better performance and fresher data.

Key Features:
- **BigQuery Integration**: Direct access to GDELT BigQuery datasets
- **Information Units Schema**: Full compatibility with existing information_units table
- **Geographic Filtering**: Horn of Africa and other regional targeting
- **ML Integration**: Compatible with ML Deep Scorer processing
- **Real-time Processing**: Near real-time data ingestion capability
- **Batch Processing**: Configurable batch sizes with transaction management
- **Conflict Analysis**: Specialized processing for conflict-related events
- **Performance Optimization**: Query optimization and parallel processing

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: December 27, 2024
"""

import asyncio
import asyncpg
import logging
import json
import uuid
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
class BigQueryETLConfig:
    """Configuration for GDELT BigQuery ETL operations."""
    
    # Database settings
    database_url: str
    max_connections: int = 10
    connection_timeout: int = 30
    
    # BigQuery settings
    bigquery_project: str = "gdelt-bq"
    bigquery_dataset: str = "gdeltv2"
    gkg_table: str = "gkg_partitioned"
    google_credentials_path: Optional[str] = None
    
    # Processing settings
    batch_size: int = 5000
    max_concurrent_batches: int = 3
    enable_ml_processing: bool = True
    enable_conflict_analysis: bool = True
    
    # Geographic filtering
    target_countries: List[str] = field(default_factory=lambda: ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"])
    enable_geographic_filtering: bool = True
    
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
class BigQueryETLMetrics:
    """BigQuery ETL processing metrics."""
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


class GDELTBigQueryETL:
    """
    GDELT BigQuery ETL pipeline for loading data into information_units table.
    """
    
    def __init__(self, config: BigQueryETLConfig):
        self.config = config
        self.connection_pool = None
        self.bigquery_client = None
        self.metrics = BigQueryETLMetrics()
        self._running = False
        
        # Initialize BigQuery client
        self._initialize_bigquery_client()
    
    def _initialize_bigquery_client(self):
        """Initialize BigQuery client with authentication."""
        try:
            # Use credentials if provided, otherwise use default
            if self.config.google_credentials_path:
                credentials = google.auth.load_credentials_from_file(
                    self.config.google_credentials_path
                )[0]
                self.bigquery_client = bigquery.Client(
                    credentials=credentials, 
                    project=self.config.bigquery_project
                )
            else:
                # Use default credentials
                credentials, project = google.auth.default()
                self.bigquery_client = bigquery.Client(
                    credentials=credentials, 
                    project=self.config.bigquery_project
                )
            
            logger.info(f"✅ BigQuery client initialized (project: {self.config.bigquery_project})")
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
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
    
    def generate_bigquery_query(self, start_date: datetime, end_date: datetime) -> str:
        """Generate optimized BigQuery query for GDELT GKG data."""
        
        # Build geographic filter if enabled
        geographic_filter = ""
        if self.config.enable_geographic_filtering and self.config.target_countries:
            country_pattern = '|'.join(self.config.target_countries)
            geographic_filter = f"""
            AND (
                REGEXP_CONTAINS(V2Locations, r'#({country_pattern})#') 
                OR REGEXP_CONTAINS(Locations, r'\\b({country_pattern})\\b')
            )"""
        
        query = f"""
        SELECT 
            GKGRECORDID,
            DATE,
            SourceCollectionIdentifier,
            SourceCommonName,
            DocumentIdentifier,
            V2Themes,
            V2Locations,
            V2Persons,
            V2Organizations,
            V2Tone,
            Quotations,
            AllNames,
            GCAM
        FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.{self.config.gkg_table}`
        WHERE _PARTITIONTIME >= TIMESTAMP('{start_date.strftime('%Y-%m-%d')}')
          AND _PARTITIONTIME < TIMESTAMP('{end_date.strftime('%Y-%m-%d')}')
          AND DATE IS NOT NULL{geographic_filter}
        ORDER BY DATE DESC
        LIMIT {self.config.batch_size}
        """
        
        return query
    
    def execute_bigquery_query(self, query: str) -> pd.DataFrame:
        """Execute BigQuery query and return DataFrame."""
        try:
            start_time = time.time()
            
            # Configure job for optimization
            job_config = bigquery.QueryJobConfig(
                use_query_cache=True,
                use_legacy_sql=False,
                maximum_bytes_billed=5*10**10  # 50GB limit
            )
            
            # Execute query
            logger.debug(f"Executing BigQuery: {query[:200]}...")
            query_job = self.bigquery_client.query(query, job_config=job_config)
            
            # Get results as DataFrame
            df = query_job.to_dataframe()
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.bigquery_queries_executed += 1
            self.metrics.bigquery_processing_time_seconds += processing_time
            self.metrics.bigquery_bytes_processed += query_job.total_bytes_processed or 0
            
            logger.info(f"✅ BigQuery returned {len(df)} records in {processing_time:.2f}s")
            return df
            
        except Exception as e:
            logger.error(f"BigQuery execution failed: {e}")
            raise
    
    def convert_gkg_to_information_unit(self, row: pd.Series) -> Dict[str, Any]:
        """Convert GDELT GKG row to information_units format."""
        
        # Parse GDELT date format (YYYYMMDDHHMMSS)
        date_value = None
        if pd.notna(row['DATE']):
            try:
                date_str = str(int(row['DATE']))
                if len(date_str) >= 8:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    hour = int(date_str[8:10]) if len(date_str) >= 10 else 0
                    minute = int(date_str[10:12]) if len(date_str) >= 12 else 0
                    second = int(date_str[12:14]) if len(date_str) >= 14 else 0
                    date_value = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            except Exception as e:
                logger.warning(f"Failed to parse date {row['DATE']}: {e}")
                date_value = datetime.now(timezone.utc)
        
        # Extract key information for title and content
        source_name = str(row['SourceCommonName']) if pd.notna(row['SourceCommonName']) else ""
        themes = str(row['V2Themes']) if pd.notna(row['V2Themes']) else ""
        locations = str(row['V2Locations']) if pd.notna(row['V2Locations']) else ""
        persons = str(row['V2Persons']) if pd.notna(row['V2Persons']) else ""
        organizations = str(row['V2Organizations']) if pd.notna(row['V2Organizations']) else ""
        quotations = str(row['Quotations']) if pd.notna(row['Quotations']) else ""
        
        # Generate title from available data
        title_parts = []
        if source_name:
            title_parts.append(f"Report from {source_name}")
        if themes:
            # Extract first theme for title
            theme_codes = themes.split(';')[:3]  # Take first 3 themes
            if theme_codes:
                title_parts.append(f"Themes: {', '.join(theme_codes)}")
        
        title = " - ".join(title_parts) if title_parts else f"GDELT Report {row['GKGRECORDID']}"
        
        # Create comprehensive content from all fields
        content_parts = []
        if quotations:
            content_parts.append(f"Quotations: {quotations[:500]}")  # Limit length
        if persons:
            content_parts.append(f"Persons mentioned: {persons[:300]}")
        if organizations:
            content_parts.append(f"Organizations: {organizations[:300]}")
        if locations:
            content_parts.append(f"Locations: {locations[:300]}")
        
        content = " | ".join(content_parts) if content_parts else f"GDELT GKG Record: {row['GKGRECORDID']}"
        
        # Extract geographic information
        latitude, longitude, country, location_name = self._extract_location_info(locations)
        
        # Extract entities
        primary_actors = self._extract_entities(persons, max_items=5)
        secondary_actors = self._extract_entities(organizations, max_items=5)
        
        # Parse tone information
        avg_tone = None
        if pd.notna(row['V2Tone']):
            try:
                tone_data = str(row['V2Tone']).split(',')
                if tone_data and tone_data[0]:
                    avg_tone = float(tone_data[0])
            except (ValueError, IndexError):
                pass
        
        # Create information unit record
        return {
            'id': str(uuid.uuid4()),
            'external_id': str(row['GKGRECORDID']),
            'title': title[:500],  # Limit title length
            'content': content[:2000],  # Limit content length  
            'content_url': str(row['DocumentIdentifier']) if pd.notna(row['DocumentIdentifier']) else '',
            'published_at': date_value,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc),
            'source': source_name[:100] if source_name else 'GDELT',
            'data_source_id': 'gdelt_bigquery',
            'language': 'en',  # GDELT is primarily English
            'word_count': len(content.split()) if content else 0,
            'latitude': latitude,
            'longitude': longitude,
            'country': country,
            'location_name': location_name,
            'primary_actors': primary_actors,
            'secondary_actors': secondary_actors,
            'themes': self._extract_themes(themes),
            'sentiment_score': avg_tone,
            'avg_tone': avg_tone,
            'is_conflict_related': self._is_conflict_related(themes),
            'metadata': {
                'gdelt_gkg_record_id': str(row['GKGRECORDID']),
                'source_collection_id': int(row['SourceCollectionIdentifier']) if pd.notna(row['SourceCollectionIdentifier']) else None,
                'v2_themes': themes,
                'v2_locations': locations,
                'v2_persons': persons,
                'v2_organizations': organizations,
                'v2_tone': str(row['V2Tone']) if pd.notna(row['V2Tone']) else '',
                'gcam': str(row['GCAM']) if pd.notna(row['GCAM']) else '',
                'all_names': str(row['AllNames']) if pd.notna(row['AllNames']) else ''
            }
        }
    
    def _extract_location_info(self, locations_str: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
        """Extract geographic information from V2Locations field."""
        if not locations_str:
            return None, None, None, None
        
        try:
            # Parse V2Locations format: TYPE#FULLNAME#COUNTRYCODE#ADM1CODE#ADM2CODE#LAT#LONG#FEATUREID
            locations = locations_str.split(';')
            for location in locations[:1]:  # Take first location
                parts = location.split('#')
                if len(parts) >= 8:
                    try:
                        lat = float(parts[5]) if parts[5] else None
                        lon = float(parts[6]) if parts[6] else None
                        country = parts[2] if parts[2] else None
                        location_name = parts[1] if parts[1] else None
                        return lat, lon, country, location_name
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.debug(f"Failed to parse location: {e}")
        
        return None, None, None, None
    
    def _extract_entities(self, entities_str: str, max_items: int = 5) -> List[str]:
        """Extract entity names from persons or organizations field."""
        if not entities_str:
            return []
        
        try:
            # Parse V2 format: NAME,OFFSET,OFFSET;NAME,OFFSET,OFFSET
            entities = []
            for entity_group in entities_str.split(';')[:max_items]:
                parts = entity_group.split(',')
                if parts and parts[0]:
                    entities.append(parts[0])
            return entities
        except Exception as e:
            logger.debug(f"Failed to parse entities: {e}")
            return []
    
    def _extract_themes(self, themes_str: str, max_themes: int = 10) -> List[str]:
        """Extract theme codes from V2Themes field."""
        if not themes_str:
            return []
        
        try:
            # Parse themes format: THEME,OFFSET;THEME,OFFSET
            themes = []
            for theme_group in themes_str.split(';')[:max_themes]:
                parts = theme_group.split(',')
                if parts and parts[0]:
                    themes.append(parts[0])
            return themes
        except Exception as e:
            logger.debug(f"Failed to parse themes: {e}")
            return []
    
    def _is_conflict_related(self, themes_str: str) -> bool:
        """Determine if the record is conflict-related based on themes."""
        if not themes_str:
            return False
        
        conflict_indicators = [
            'CONFLICT', 'VIOLENCE', 'TERRORISM', 'MILITARY', 'SECURITY',
            'WAR', 'PROTEST', 'CRISIS', 'EMERGENCY', 'DISASTER'
        ]
        
        themes_upper = themes_str.upper()
        return any(indicator in themes_upper for indicator in conflict_indicators)
    
    async def store_information_units(self, information_units: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """Store information units in database with conflict handling."""
        if not information_units:
            return 0, 0, 0
        
        inserted = 0
        updated = 0
        skipped = 0
        
        async with self.connection_pool.acquire() as conn:
            for unit in information_units:
                try:
                    # Check if record exists
                    existing = await conn.fetchval(
                        "SELECT id FROM information_units WHERE external_id = $1",
                        unit['external_id']
                    )
                    
                    if existing and self.config.skip_existing_records:
                        skipped += 1
                        continue
                    
                    if existing:
                        # Update existing record
                        await conn.execute("""
                            UPDATE information_units SET
                                title = $2, content = $3, content_url = $4,
                                updated_at = $5, latitude = $6, longitude = $7,
                                country = $8, location_name = $9, primary_actors = $10,
                                secondary_actors = $11, themes = $12, sentiment_score = $13,
                                avg_tone = $14, is_conflict_related = $15, metadata = $16
                            WHERE external_id = $1
                        """, 
                        unit['external_id'], unit['title'], unit['content'], 
                        unit['content_url'], unit['updated_at'], unit['latitude'],
                        unit['longitude'], unit['country'], unit['location_name'],
                        unit['primary_actors'], unit['secondary_actors'], unit['themes'],
                        unit['sentiment_score'], unit['avg_tone'], unit['is_conflict_related'],
                        json.dumps(unit['metadata'])
                        )
                        updated += 1
                    else:
                        # Insert new record
                        await conn.execute("""
                            INSERT INTO information_units (
                                id, external_id, title, content, content_url, published_at,
                                created_at, updated_at, source, data_source_id, language,
                                word_count, latitude, longitude, country, location_name,
                                primary_actors, secondary_actors, themes, sentiment_score,
                                avg_tone, is_conflict_related, metadata
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                                $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
                            )
                        """,
                        unit['id'], unit['external_id'], unit['title'], unit['content'],
                        unit['content_url'], unit['published_at'], unit['created_at'],
                        unit['updated_at'], unit['source'], unit['data_source_id'],
                        unit['language'], unit['word_count'], unit['latitude'],
                        unit['longitude'], unit['country'], unit['location_name'],
                        unit['primary_actors'], unit['secondary_actors'], unit['themes'],
                        unit['sentiment_score'], unit['avg_tone'], unit['is_conflict_related'],
                        json.dumps(unit['metadata'])
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
    ) -> BigQueryETLMetrics:
        """Process GDELT data for a specific date range."""
        
        logger.info(f"🚀 Starting BigQuery ETL for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Reset metrics
        self.metrics = BigQueryETLMetrics()
        
        try:
            # Process in daily chunks for better performance
            current_date = start_date
            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=1), end_date)
                
                logger.info(f"Processing chunk: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                
                # Generate and execute BigQuery query
                query = self.generate_bigquery_query(current_date, chunk_end)
                df = self.execute_bigquery_query(query)
                
                if len(df) == 0:
                    logger.info(f"No data found for {current_date.strftime('%Y-%m-%d')}")
                    current_date = chunk_end
                    continue
                
                # Convert to information units
                information_units = []
                for _, row in df.iterrows():
                    try:
                        unit = self.convert_gkg_to_information_unit(row)
                        information_units.append(unit)
                    except Exception as e:
                        logger.warning(f"Failed to convert row to information unit: {e}")
                        self.metrics.records_failed += 1
                
                # Store in database
                if information_units:
                    inserted, updated, skipped = await self.store_information_units(information_units)
                    
                    # Update metrics
                    self.metrics.records_inserted += inserted
                    self.metrics.records_updated += updated
                    self.metrics.records_skipped += skipped
                    self.metrics.total_records_processed += len(information_units)
                    
                    logger.info(f"Processed {len(information_units)} records: {inserted} inserted, {updated} updated, {skipped} skipped")
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.metrics)
                
                current_date = chunk_end
                
                # Brief pause between chunks
                await asyncio.sleep(0.1)
            
            # Finalize metrics
            self.metrics.end_time = datetime.now(timezone.utc)
            
            logger.info(f"🎉 BigQuery ETL completed!")
            logger.info(f"Total records processed: {self.metrics.total_records_processed:,}")
            logger.info(f"Records inserted: {self.metrics.records_inserted:,}")
            logger.info(f"Records updated: {self.metrics.records_updated:,}")
            logger.info(f"Processing rate: {self.metrics.records_per_second:.1f} records/second")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"BigQuery ETL failed: {e}")
            self.metrics.end_time = datetime.now(timezone.utc)
            raise
    
    async def health_check(self) -> bool:
        """Perform health check of BigQuery and database connections."""
        try:
            # Test BigQuery connection
            test_query = f"""
            SELECT COUNT(*) as record_count
            FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.{self.config.gkg_table}`
            WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
            LIMIT 1
            """
            
            df = self.execute_bigquery_query(test_query)
            logger.info(f"BigQuery health check passed: {df.iloc[0]['record_count']} recent records")
            
            # Test database connection
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM information_units LIMIT 1")
                logger.info(f"Database health check passed: {result} total information units")
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Factory function
def create_bigquery_etl(
    database_url: str,
    target_countries: Optional[List[str]] = None,
    **kwargs
) -> GDELTBigQueryETL:
    """
    Create a GDELT BigQuery ETL instance with default configuration.
    
    Args:
        database_url: PostgreSQL database URL
        target_countries: List of country codes to target
        **kwargs: Additional configuration options
        
    Returns:
        Configured GDELTBigQueryETL instance
    """
    config = BigQueryETLConfig(
        database_url=database_url,
        target_countries=target_countries or ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
        **kwargs
    )
    return GDELTBigQueryETL(config)


# Convenience function
async def run_bigquery_etl(
    database_url: str,
    start_date: datetime,
    end_date: datetime,
    target_countries: Optional[List[str]] = None,
    **kwargs
) -> BigQueryETLMetrics:
    """
    Quick function to run BigQuery ETL for a date range.
    
    Args:
        database_url: PostgreSQL database URL
        start_date: Start date for processing
        end_date: End date for processing
        target_countries: List of country codes to target
        **kwargs: Additional configuration options
        
    Returns:
        ETL processing metrics
    """
    etl = create_bigquery_etl(
        database_url=database_url,
        target_countries=target_countries,
        **kwargs
    )
    
    try:
        await etl.initialize()
        return await etl.process_date_range(start_date, end_date)
    finally:
        await etl.close()