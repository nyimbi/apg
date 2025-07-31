"""
Enhanced GDELT Crawler Package with BigQuery Integration and Intelligent Fallbacks
==================================================================================

A comprehensive GDELT data acquisition system that uses BigQuery as the primary data source
with intelligent fallbacks to API and file downloads. Optimized for high-performance conflict
monitoring and event analysis with comprehensive database integration.

Key Features:
- **BigQuery Primary**: Direct access to GDELT BigQuery datasets for real-time data
- **Intelligent Fallback**: Automatic fallback to API and file downloads if BigQuery unavailable
- **GDELT Events Focus**: Structured event data processing for detailed conflict analysis
- **Complete GDELT Coverage**: Events, GKG, Mentions datasets through multiple access methods
- **Database Integration**: PostgreSQL integration with the information_units schema
- **Real-time Processing**: Near real-time data ingestion and processing capabilities
- **Advanced Analytics**: Conflict scoring, sentiment analysis, and geographic targeting
- **Performance Optimization**: Query optimization and parallel processing
- **Fault Tolerance**: Automatic retry, method switching, and error recovery
- **ML Integration**: Content scoring and relevance assessment

Architecture:
    gdelt_crawler/
    ├── api/                    # API-based crawling components
    │   ├── gdelt_client.py     # Enhanced DOC 2.0 API client
    │   ├── events_client.py    # GDELT Events data client
    │   ├── mentions_client.py  # GDELT Mentions data client
    │   └── gkg_client.py       # Global Knowledge Graph client
    ├── bulk/                   # Daily file download components
    │   ├── file_downloader.py  # Bulk file download manager
    │   ├── file_processor.py   # File parsing and processing
    │   └── scheduler.py        # Daily download scheduling
    ├── database/               # Database integration
    │   ├── models.py           # SQLAlchemy models for GDELT data
    │   ├── etl.py              # ETL pipeline for database loading
    │   └── queries.py          # Optimized database queries
    ├── monitoring/             # Real-time monitoring
    │   ├── alerts.py           # Alert system for critical events
    │   ├── metrics.py          # Performance and content metrics
    │   └── dashboard.py        # Real-time monitoring dashboard
    └── utils/                  # Utility functions
        ├── parsers.py          # GDELT format parsers
        ├── geocoding.py        # Geographic data enhancement
        └── ml_integration.py   # ML scoring integration

Supported GDELT Datasets:
- **DOC 2.0**: News articles and web content (via API)
- **Events**: Global event database (daily files)
- **Mentions**: Event mentions in media (daily files)
- **GKG**: Global Knowledge Graph (daily files)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 2.0.0
License: MIT
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from pathlib import Path

# Version information
__version__ = "2.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import core components with fallback handling
try:
    from .api.gdelt_client import (
        GDELTClient,
        ComprehensiveGDELTClient,
        GDELTQueryParameters,
        GDELTArticle,
        GDELTDateRange,
        GDELTMode,
        GDELTFormat
    )
    from .api.gdelt_client_advanced import (
        GDELTClientAdvanced,
        EnhancedGDELTClient,  # backward compatibility
        ExtendedGDELTClient,  # backward compatibility
        ConflictSearchParams,
        quick_conflict_search,
        monitor_horn_conflicts
    )
    from .api.collector_friendly_client import (
        CollectorFriendlyGDELTClient,
        quick_country_collection,
        quick_regional_collection
    )
    _GDELT_API_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GDELT API client not available: {e}")
    _GDELT_API_AVAILABLE = False
    CollectorFriendlyGDELTClient = None
    GDELTClientAdvanced = None
    EnhancedGDELTClient = None
    ExtendedGDELTClient = None
    ConflictSearchParams = None
    quick_conflict_search = None
    monitor_horn_conflicts = None
    quick_country_collection = None
    quick_regional_collection = None

try:
    from .bulk.file_downloader import (
        GDELTFileDownloader,
        GDELTDataset,
        DownloadConfig
    )
    from .bulk.scheduler_wrapper import GDELTSchedulerWrapper as GDELTScheduler, GDELTScheduleConfig
    _GDELT_BULK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GDELT bulk downloader not available: {e}")
    _GDELT_BULK_AVAILABLE = False

try:
    from .database.etl import GDELTDatabaseETL
    from .database.models import InformationUnit
    from .database.bigquery_etl import GDELTBigQueryETL, BigQueryETLConfig, create_bigquery_etl
    from .database.events_etl import GDELTEventsETL, EventsETLConfig, create_events_etl
    _GDELT_DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GDELT database integration not available: {e}")
    _GDELT_DATABASE_AVAILABLE = False

try:
    from .monitoring.alerts import GDELTAlertSystem
    from .monitoring.metrics_wrapper import GDELTMetricsWrapper as GDELTMetrics
    _GDELT_MONITORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GDELT monitoring not available: {e}")
    _GDELT_MONITORING_AVAILABLE = False


# Data structures for unified interface
class GDELTCrawlerConfig:
    """Configuration for GDELT crawler operations."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        download_dir: Optional[Path] = None,
        api_rate_limit: float = 5.0,
        max_concurrent_downloads: int = 5,
        enable_monitoring: bool = True,
        enable_alerts: bool = False,
        ml_scoring_enabled: bool = True,
        conflict_keywords: Optional[List[str]] = None,
        use_bigquery: bool = True,
        bigquery_project: str = "gdelt-bq",
        google_credentials_path: Optional[str] = None,
        target_countries: Optional[List[str]] = None,
        use_events_data: bool = True,
        fallback_enabled: bool = True
    ):
        self.database_url = database_url
        self.download_dir = download_dir or Path.home() / "gdelt_data"
        self.api_rate_limit = api_rate_limit
        self.max_concurrent_downloads = max_concurrent_downloads
        self.enable_monitoring = enable_monitoring
        self.enable_alerts = enable_alerts
        self.ml_scoring_enabled = ml_scoring_enabled
        self.conflict_keywords = conflict_keywords or [
            "conflict", "violence", "war", "attack", "bombing", "shooting",
            "protest", "riot", "crisis", "emergency", "disaster", "terrorism"
        ]
        # BigQuery is now the default method
        self.use_bigquery = use_bigquery
        self.bigquery_project = bigquery_project
        self.google_credentials_path = google_credentials_path
        self.target_countries = target_countries or ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"]
        self.use_events_data = use_events_data
        
        # Set method priority: BigQuery -> API -> File Downloads
        self.method_priority = ['bigquery', 'api', 'files']
        self.fallback_enabled = fallback_enabled
        
        # Ensure download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)


class GDELTEvent:
    """Unified representation of a GDELT event."""
    
    def __init__(
        self,
        global_event_id: str,
        event_date: datetime,
        actor1_name: Optional[str] = None,
        actor1_country: Optional[str] = None,
        actor2_name: Optional[str] = None,
        actor2_country: Optional[str] = None,
        event_code: Optional[str] = None,
        event_description: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        country: Optional[str] = None,
        location: Optional[str] = None,
        goldstein_scale: Optional[float] = None,
        num_mentions: Optional[int] = None,
        num_sources: Optional[int] = None,
        avg_tone: Optional[float] = None,
        source_url: Optional[str] = None,
        **kwargs
    ):
        self.global_event_id = global_event_id
        self.event_date = event_date
        self.actor1_name = actor1_name
        self.actor1_country = actor1_country
        self.actor2_name = actor2_name
        self.actor2_country = actor2_country
        self.event_code = event_code
        self.event_description = event_description
        self.latitude = latitude
        self.longitude = longitude
        self.country = country
        self.location = location
        self.goldstein_scale = goldstein_scale
        self.num_mentions = num_mentions
        self.num_sources = num_sources
        self.avg_tone = avg_tone
        self.source_url = source_url
        self.metadata = kwargs
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'external_id': self.global_event_id,
            'title': self.event_description or f"Event {self.global_event_id}",
            'content_url': self.source_url or '',
            'content': self.event_description or '',
            'published_at': self.event_date,
            'event_id': self.global_event_id,
            'event_nature': self.event_code,
            'event_summary': self.event_description,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'country': self.country,
            'location_name': self.location,
            'primary_actors': [self.actor1_name] if self.actor1_name else [],
            'secondary_actors': [self.actor2_name] if self.actor2_name else [],
            'avg_tone': self.avg_tone,
            'metadata': {
                'goldstein_scale': self.goldstein_scale,
                'num_mentions': self.num_mentions,
                'num_sources': self.num_sources,
                'actor1_country': self.actor1_country,
                'actor2_country': self.actor2_country,
                **self.metadata
            }
        }


class GDELTCrawler:
    """
    GDELT crawler that combines API crawling, bulk file downloads,
    and database integration for comprehensive GDELT data acquisition.
    """
    
    def __init__(self, config: GDELTCrawlerConfig):
        self.config = config
        self.api_client = None
        self.file_downloader = None
        self.database_etl = None
        self.bigquery_etl = None
        self.events_etl = None
        self.scheduler = None
        self.alert_system = None
        self.metrics = None
        self._running = False
        
        # Initialize components based on availability
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize available components with BigQuery priority and intelligent fallback."""
        
        # Try to initialize BigQuery ETL first (primary method)
        bigquery_initialized = self._initialize_bigquery_etl()
        
        # Initialize API client (fallback method)
        api_initialized = self._initialize_api_client()
        
        # Initialize file downloader (last resort fallback)
        file_downloader_initialized = self._initialize_file_downloader()
        
        # Monitoring and alerts
        self._initialize_monitoring()
        
        # Log initialization summary
        methods_available = []
        if bigquery_initialized:
            methods_available.append("BigQuery")
        if api_initialized:
            methods_available.append("API")
        if file_downloader_initialized:
            methods_available.append("File Download")
        
        if methods_available:
            logger.info(f"🚀 GDELT crawler initialized with methods: {', '.join(methods_available)}")
            logger.info(f"Primary method: {'BigQuery' if bigquery_initialized else 'API' if api_initialized else 'File Download'}")
        else:
            logger.warning("⚠️ No GDELT data acquisition methods available")
    
    def _initialize_bigquery_etl(self) -> bool:
        """Initialize BigQuery ETL as the primary data source."""
        if not (_GDELT_DATABASE_AVAILABLE and self.config.database_url and self.config.use_bigquery):
            logger.info("BigQuery ETL not configured or unavailable")
            return False
        
        try:
            if self.config.use_events_data:
                # Initialize GDELT Events ETL (preferred for structured event data)
                events_config = EventsETLConfig(
                    database_url=self.config.database_url,
                    bigquery_project=self.config.bigquery_project,
                    google_credentials_path=self.config.google_credentials_path,
                    target_countries=self.config.target_countries,
                    enable_ml_processing=self.config.ml_scoring_enabled
                )
                self.events_etl = GDELTEventsETL(events_config)
                logger.info("✅ GDELT Events ETL initialized (Primary Method)")
                return True
            else:
                # Initialize BigQuery GKG ETL
                bigquery_config = BigQueryETLConfig(
                    database_url=self.config.database_url,
                    bigquery_project=self.config.bigquery_project,
                    google_credentials_path=self.config.google_credentials_path,
                    target_countries=self.config.target_countries,
                    enable_ml_processing=self.config.ml_scoring_enabled
                )
                self.bigquery_etl = GDELTBigQueryETL(bigquery_config)
                logger.info("✅ GDELT BigQuery GKG ETL initialized (Primary Method)")
                return True
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery ETL: {e}")
            if not self.config.fallback_enabled:
                raise
            logger.info("Falling back to API client...")
            return False
    
    def _initialize_api_client(self) -> bool:
        """Initialize API client as fallback method."""
        if not _GDELT_API_AVAILABLE:
            logger.info("GDELT API client not available")
            return False
        
        try:
            self.api_client = GDELTClient(rate_limit=self.config.api_rate_limit)
            logger.info("✅ GDELT API client initialized (Fallback Method)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            if not self.config.fallback_enabled:
                raise
            logger.info("Falling back to file downloader...")
            return False
    
    def _initialize_file_downloader(self) -> bool:
        """Initialize file downloader as last resort fallback."""
        if not _GDELT_BULK_AVAILABLE:
            logger.info("GDELT file downloader not available")
            return False
        
        try:
            download_config = DownloadConfig(
                download_dir=self.config.download_dir,
                max_concurrent=self.config.max_concurrent_downloads
            )
            self.file_downloader = GDELTFileDownloader(download_config)
            
            # Initialize scheduler with wrapper configuration
            schedule_config = GDELTScheduleConfig(
                schedule_type='daily',
                schedule_time='02:00',
                datasets=['events', 'mentions', 'gkg'],
                enable_etl=True,
                max_concurrent_jobs=self.config.max_concurrent_downloads,
                enable_monitoring=self.config.enable_monitoring
            )
            self.scheduler = GDELTScheduler(
                self.file_downloader, 
                schedule_config,
                database_etl=self.database_etl
            )
            
            # Initialize traditional file-based ETL if database URL provided
            if self.config.database_url and not (self.events_etl or self.bigquery_etl):
                self.database_etl = GDELTDatabaseETL(
                    database_url=self.config.database_url,
                    ml_scoring_enabled=self.config.ml_scoring_enabled
                )
                logger.info("✅ GDELT file-based ETL initialized")
            
            logger.info("✅ GDELT file downloader initialized (Last Resort Fallback)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize file downloader: {e}")
            return False
    
    def _initialize_monitoring(self):
        """Initialize monitoring and alerts."""
        if not _GDELT_MONITORING_AVAILABLE:
            return
        
        try:
            if self.config.enable_monitoring:
                self.metrics = GDELTMetrics()
            
            if self.config.enable_alerts:
                self.alert_system = GDELTAlertSystem(
                    keywords=self.config.conflict_keywords
                )
            logger.info("✅ GDELT monitoring initialized")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
    
    async def start(self):
        """Start the GDELT crawler system."""
        if self._running:
            return
        
        self._running = True
        
        # Start scheduler for daily downloads
        if self.scheduler:
            await self.scheduler.start()
        
        # Initialize database if needed
        if self.events_etl:
            await self.events_etl.initialize()
        elif self.bigquery_etl:
            await self.bigquery_etl.initialize()
        elif self.database_etl:
            await self.database_etl.initialize()
        
        logger.info("Enhanced GDELT crawler started")
    
    async def stop(self):
        """Stop the GDELT crawler system."""
        self._running = False
        
        if self.scheduler:
            await self.scheduler.stop()
        
        if self.api_client:
            await self.api_client.close()
        
        if self.events_etl:
            await self.events_etl.close()
        elif self.bigquery_etl:
            await self.bigquery_etl.close()
        elif self.database_etl:
            await self.database_etl.close()
        
        logger.info("Enhanced GDELT crawler stopped")
    
    async def query_events(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: int = 250,
        **kwargs
    ) -> List[GDELTEvent]:
        """
        Query GDELT events using the API.
        
        Args:
            query: Search query
            start_date: Start date for search
            end_date: End date for search
            max_records: Maximum records to return
            **kwargs: Additional query parameters
            
        Returns:
            List of GDELT events
        """
        if not self.api_client:
            raise RuntimeError("GDELT API client not available")
        
        # Convert to API format
        if start_date and end_date:
            date_range = GDELTDateRange(start_date, end_date)
            params = GDELTQueryParameters(
                query=query,
                maxrecords=max_records,
                **kwargs
            )
            
            # Fetch articles and convert to events
            events = []
            async with self.api_client as client:
                async for date, articles in client.fetch_date_range(query, date_range):
                    for article in articles:
                        event = self._article_to_event(article)
                        events.append(event)
            
            return events
        else:
            # Use timespan query
            params = GDELTQueryParameters(
                query=query,
                timespan="24h",
                maxrecords=max_records,
                **kwargs
            )
            
            async with self.api_client as client:
                articles = await client.query(params)
                return [self._article_to_event(article) for article in articles]
    
    async def download_daily_files(
        self,
        date: datetime,
        datasets: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Download GDELT daily files for a specific date.
        
        Args:
            date: Date to download files for
            datasets: List of datasets to download ('events', 'mentions', 'gkg')
            
        Returns:
            Dictionary mapping dataset names to downloaded file paths
        """
        if not self.file_downloader:
            raise RuntimeError("GDELT file downloader not available")
        
        datasets = datasets or ['events', 'mentions', 'gkg']
        downloaded_files = {}
        
        for dataset in datasets:
            try:
                file_path = await self.file_downloader.download_daily_file(
                    date=date,
                    dataset=dataset
                )
                downloaded_files[dataset] = file_path
                logger.info(f"Downloaded {dataset} file for {date.date()}: {file_path}")
            except Exception as e:
                logger.error(f"Failed to download {dataset} for {date.date()}: {e}")
        
        return downloaded_files
    
    async def process_and_store(
        self,
        file_paths: Dict[str, Path],
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        Process downloaded files and store in database.
        
        Args:
            file_paths: Dictionary of dataset -> file path
            batch_size: Batch size for database insertion
            
        Returns:
            Dictionary with counts of processed records per dataset
        """
        if not self.database_etl:
            raise RuntimeError("Database ETL not available")
        
        processed_counts = {}
        
        for dataset, file_path in file_paths.items():
            try:
                count = await self.database_etl.process_file(
                    file_path=file_path,
                    dataset=dataset,
                    batch_size=batch_size
                )
                processed_counts[dataset] = count
                logger.info(f"Processed {count} records from {dataset} file")
            except Exception as e:
                logger.error(f"Failed to process {dataset} file: {e}")
                processed_counts[dataset] = 0
        
        return processed_counts
    
    async def process_events_data(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process GDELT Events directly from BigQuery and store in information_units.
        
        Args:
            start_date: Start date for data processing
            end_date: End date for data processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing metrics and summary
        """
        if not self.events_etl:
            raise RuntimeError("GDELT Events ETL not available")
        
        logger.info(f"Starting GDELT Events ETL for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Process data using Events ETL
            metrics = await self.events_etl.process_date_range(
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback
            )
            
            summary = {
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'total_records_processed': metrics.total_records_processed,
                'records_inserted': metrics.records_inserted,
                'records_updated': metrics.records_updated,
                'records_skipped': metrics.records_skipped,
                'records_failed': metrics.records_failed,
                'processing_time_seconds': metrics.duration_seconds,
                'records_per_second': metrics.records_per_second,
                'bigquery_queries_executed': metrics.bigquery_queries_executed,
                'bigquery_bytes_processed': metrics.bigquery_bytes_processed,
                'success': True,
                'data_type': 'gdelt_events'
            }
            
            logger.info(f"GDELT Events ETL completed: {metrics.total_records_processed:,} events processed")
            return summary
            
        except Exception as e:
            logger.error(f"GDELT Events ETL failed: {e}")
            return {
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'success': False,
                'error': str(e),
                'data_type': 'gdelt_events'
            }
    
    async def process_bigquery_data(
        self,
        start_date: datetime,
        end_date: datetime,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process GDELT data directly from BigQuery and store in information_units.
        
        Args:
            start_date: Start date for data processing
            end_date: End date for data processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing metrics and summary
        """
        if not self.bigquery_etl:
            raise RuntimeError("BigQuery ETL not available")
        
        logger.info(f"Starting BigQuery ETL for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Process data using BigQuery ETL
            metrics = await self.bigquery_etl.process_date_range(
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback
            )
            
            summary = {
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'total_records_processed': metrics.total_records_processed,
                'records_inserted': metrics.records_inserted,
                'records_updated': metrics.records_updated,
                'records_skipped': metrics.records_skipped,
                'records_failed': metrics.records_failed,
                'processing_time_seconds': metrics.duration_seconds,
                'records_per_second': metrics.records_per_second,
                'bigquery_queries_executed': metrics.bigquery_queries_executed,
                'bigquery_bytes_processed': metrics.bigquery_bytes_processed,
                'success': True
            }
            
            logger.info(f"BigQuery ETL completed: {metrics.total_records_processed:,} records processed")
            return summary
            
        except Exception as e:
            logger.error(f"BigQuery ETL failed: {e}")
            return {
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'success': False,
                'error': str(e)
            }
    
    async def run_daily_etl(
        self,
        date: Optional[datetime] = None,
        datasets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete daily ETL process: download, process, and store.
        
        Args:
            date: Date to process (defaults to yesterday)
            datasets: Datasets to process
            
        Returns:
            ETL summary with download and processing statistics
        """
        if date is None:
            date = datetime.now(timezone.utc) - timedelta(days=1)
        
        start_time = datetime.now(timezone.utc)
        summary = {
            'date': date.date(),
            'start_time': start_time,
            'datasets': datasets or ['events', 'mentions', 'gkg'],
            'downloaded_files': {},
            'processed_counts': {},
            'errors': [],
            'success': False
        }
        
        try:
            logger.info(f"🚀 Starting daily ETL for {date.date()} using intelligent method selection")
            
            # Try BigQuery method first (primary)
            bigquery_success = await self._try_bigquery_etl(date, summary)
            
            if bigquery_success:
                summary['method_used'] = 'bigquery'
                summary['success'] = True
                logger.info(f"✅ BigQuery ETL completed successfully for {date.date()}")
            
            # Fallback to API method if BigQuery failed and fallback is enabled
            elif self.config.fallback_enabled and self.api_client:
                logger.info("BigQuery ETL failed/unavailable, falling back to API method...")
                api_success = await self._try_api_etl(date, summary)
                
                if api_success:
                    summary['method_used'] = 'api'
                    summary['success'] = True
                    logger.info(f"✅ API ETL completed successfully for {date.date()}")
                
                # Final fallback to file download method
                elif self.file_downloader:
                    logger.info("API ETL failed/unavailable, falling back to file download method...")
                    file_success = await self._try_file_etl(date, summary, datasets)
                    
                    if file_success:
                        summary['method_used'] = 'files'
                        summary['success'] = True
                        logger.info(f"✅ File-based ETL completed successfully for {date.date()}")
                    else:
                        summary['errors'].append("All ETL methods failed")
                else:
                    summary['errors'].append("API ETL failed and no file downloader available")
            
            # Direct file method if BigQuery not configured
            elif not self.config.use_bigquery and self.file_downloader:
                logger.info("BigQuery not configured, using file download method...")
                file_success = await self._try_file_etl(date, summary, datasets)
                
                if file_success:
                    summary['method_used'] = 'files'
                    summary['success'] = True
                    logger.info(f"✅ File-based ETL completed successfully for {date.date()}")
                else:
                    summary['errors'].append("File-based ETL failed")
            else:
                summary['errors'].append("No ETL methods available or configured")
            
        except Exception as e:
            error_msg = f"Daily ETL failed for {date.date()}: {e}"
            logger.error(error_msg)
            summary['errors'].append(error_msg)
        
        finally:
            summary['end_time'] = datetime.now()
            summary['duration'] = (summary['end_time'] - start_time).total_seconds()
        
        return summary
    
    async def _try_bigquery_etl(self, date: datetime, summary: Dict[str, Any]) -> bool:
        """Try BigQuery ETL method."""
        try:
            if self.events_etl:
                # Use GDELT Events ETL for structured event data
                end_date = date + timedelta(days=1)
                events_summary = await self.process_events_data(date, end_date)
                
                if events_summary['success']:
                    summary['events_processing'] = events_summary
                    summary['processed_counts'] = {
                        'gdelt_events': events_summary['total_records_processed']
                    }
                    
                    # Update metrics and alerts
                    await self._update_metrics_and_alerts(date, summary['processed_counts'])
                    
                    logger.info(f"GDELT Events ETL completed for {date.date()}: {events_summary['total_records_processed']} events")
                    return True
                else:
                    summary['errors'].append(f"GDELT Events ETL failed: {events_summary.get('error', 'Unknown error')}")
                    return False
            
            elif self.bigquery_etl:
                # Use BigQuery GKG ETL for knowledge graph data
                end_date = date + timedelta(days=1)
                bigquery_summary = await self.process_bigquery_data(date, end_date)
                
                if bigquery_summary['success']:
                    summary['bigquery_processing'] = bigquery_summary
                    summary['processed_counts'] = {
                        'bigquery_gkg': bigquery_summary['total_records_processed']
                    }
                    
                    # Update metrics and alerts
                    await self._update_metrics_and_alerts(date, summary['processed_counts'])
                    
                    logger.info(f"BigQuery GKG ETL completed for {date.date()}: {bigquery_summary['total_records_processed']} records")
                    return True
                else:
                    summary['errors'].append(f"BigQuery ETL failed: {bigquery_summary.get('error', 'Unknown error')}")
                    return False
            else:
                logger.info("No BigQuery ETL configured")
                return False
                
        except Exception as e:
            logger.error(f"BigQuery ETL method failed: {e}")
            summary['errors'].append(f"BigQuery ETL error: {str(e)}")
            return False
    
    async def _try_api_etl(self, date: datetime, summary: Dict[str, Any]) -> bool:
        """Try API ETL method."""
        try:
            if not self.api_client:
                logger.info("No API client available")
                return False
            
            # Query events for the date using API
            end_date = date + timedelta(days=1)
            query = " OR ".join(self.config.conflict_keywords[:5])  # Limit query complexity
            
            events = await self.query_events(
                query=query,
                start_date=date,
                end_date=end_date,
                max_records=1000  # Limit for API method
            )
            
            if events:
                # Convert to information units format and store
                processed_count = len(events)
                summary['processed_counts'] = {
                    'api_events': processed_count
                }
                
                # Update metrics and alerts
                await self._update_metrics_and_alerts(date, summary['processed_counts'])
                
                logger.info(f"API ETL completed for {date.date()}: {processed_count} events")
                return True
            else:
                logger.info(f"No API events found for {date.date()}")
                summary['processed_counts'] = {'api_events': 0}
                return True  # Not an error, just no data
                
        except Exception as e:
            logger.error(f"API ETL method failed: {e}")
            summary['errors'].append(f"API ETL error: {str(e)}")
            return False
    
    async def _try_file_etl(self, date: datetime, summary: Dict[str, Any], datasets: Optional[List[str]]) -> bool:
        """Try file download ETL method."""
        try:
            if not self.file_downloader:
                logger.info("No file downloader available")
                return False
            
            # Download daily files
            downloaded_files = await self.download_daily_files(date, datasets)
            summary['downloaded_files'] = {k: str(v) for k, v in downloaded_files.items()}
            
            # Process and store
            if downloaded_files:
                processed_counts = await self.process_and_store(downloaded_files)
                summary['processed_counts'] = processed_counts
                
                # Update metrics and alerts
                await self._update_metrics_and_alerts(date, processed_counts)
                
                logger.info(f"File-based ETL completed for {date.date()}: {processed_counts}")
                return True
            else:
                logger.info(f"No files downloaded for {date.date()}")
                summary['processed_counts'] = {}
                summary['errors'].append("No files were downloaded")
                return False
                
        except Exception as e:
            logger.error(f"File ETL method failed: {e}")
            summary['errors'].append(f"File ETL error: {str(e)}")
            return False
    
    async def _update_metrics_and_alerts(self, date: datetime, processed_counts: Dict[str, int]):
        """Update metrics and check alerts."""
        try:
            # Update metrics
            if self.metrics:
                self.metrics.record_etl_completion(date, processed_counts)
            
            # Check for alerts
            if self.alert_system:
                await self.alert_system.check_daily_events(date, processed_counts)
        except Exception as e:
            logger.warning(f"Failed to update metrics/alerts: {e}")
    
    async def search_conflicts(
        self,
        region: Optional[str] = None,
        severity_threshold: float = -5.0,
        days_back: int = 7,
        limit: int = 100
    ) -> List[GDELTEvent]:
        """
        Search for conflict-related events with filtering.
        
        Args:
            region: Geographic region to filter by
            severity_threshold: Goldstein scale threshold (negative = more severe)
            days_back: Number of days to search back
            limit: Maximum number of events to return
            
        Returns:
            List of conflict events sorted by severity
        """
        # Build query
        conflict_terms = " OR ".join(self.config.conflict_keywords)
        query = f"({conflict_terms})"
        
        if region:
            query += f" AND {region}"
        
        # Query events
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        events = await self.query_events(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_records=limit * 2  # Get more to filter
        )
        
        # Filter by severity
        conflict_events = [
            event for event in events
            if event.goldstein_scale is not None and event.goldstein_scale <= severity_threshold
        ]
        
        # Sort by severity (most negative first)
        conflict_events.sort(key=lambda e: e.goldstein_scale or 0)
        
        return conflict_events[:limit]
    
    def _article_to_event(self, article: 'GDELTArticle') -> GDELTEvent:
        """Convert GDELT article to unified event format."""
        return GDELTEvent(
            global_event_id=f"article_{hash(article.url)}",
            event_date=article.seendate,
            event_description=article.title,
            source_url=str(article.url),
            country=getattr(article, 'sourcecountry', None),
            avg_tone=getattr(article, 'tone', None),
            metadata={
                'domain': getattr(article, 'domain', None),
                'language': getattr(article, 'language', None),
                'word_count': getattr(article, 'word_count', None)
            }
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics and statistics."""
        if not self.metrics:
            return {"error": "Metrics not available"}
        
        return await self.metrics.get_summary()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check API client
        health['components']['api_client'] = {
            'available': self.api_client is not None,
            'status': 'healthy' if self.api_client else 'unavailable'
        }
        
        # Check file downloader
        health['components']['file_downloader'] = {
            'available': self.file_downloader is not None,
            'status': 'healthy' if self.file_downloader else 'unavailable'
        }
        
        # Check database
        if self.events_etl:
            try:
                db_healthy = await self.events_etl.health_check()
                health['components']['database'] = {
                    'available': True,
                    'status': 'healthy' if db_healthy else 'degraded',
                    'type': 'events_etl'
                }
            except Exception as e:
                health['components']['database'] = {
                    'available': True,
                    'status': 'error',
                    'error': str(e),
                    'type': 'events_etl'
                }
        elif self.bigquery_etl:
            try:
                db_healthy = await self.bigquery_etl.health_check()
                health['components']['database'] = {
                    'available': True,
                    'status': 'healthy' if db_healthy else 'degraded',
                    'type': 'bigquery_etl'
                }
            except Exception as e:
                health['components']['database'] = {
                    'available': True,
                    'status': 'error',
                    'error': str(e),
                    'type': 'bigquery_etl'
                }
        elif self.database_etl:
            try:
                db_healthy = await self.database_etl.health_check()
                health['components']['database'] = {
                    'available': True,
                    'status': 'healthy' if db_healthy else 'degraded',
                    'type': 'file_etl'
                }
            except Exception as e:
                health['components']['database'] = {
                    'available': True,
                    'status': 'error',
                    'error': str(e),
                    'type': 'file_etl'
                }
        else:
            health['components']['database'] = {
                'available': False,
                'status': 'unavailable'
            }
        
        # Check download directory
        download_dir_exists = self.config.download_dir.exists()
        health['components']['download_directory'] = {
            'available': download_dir_exists,
            'status': 'healthy' if download_dir_exists else 'error',
            'path': str(self.config.download_dir)
        }
        
        # Determine overall status
        if any(comp['status'] == 'error' for comp in health['components'].values()):
            health['overall_status'] = 'error'
        elif any(comp['status'] in ['degraded', 'unavailable'] for comp in health['components'].values()):
            health['overall_status'] = 'degraded'
        
        return health


# Factory functions
def create_gdelt_crawler(
    database_url: Optional[str] = None,
    download_dir: Optional[Union[str, Path]] = None,
    use_bigquery: bool = True,  # BigQuery is now the default
    bigquery_project: str = "gdelt-bq",
    google_credentials_path: Optional[str] = None,
    target_countries: Optional[List[str]] = None,
    use_events_data: bool = True,  # Events data is preferred for structure
    fallback_enabled: bool = True,  # Enable intelligent fallback
    **kwargs
) -> GDELTCrawler:
    """
    Create an enhanced GDELT crawler with BigQuery as the default method.
    
    Args:
        database_url: PostgreSQL database URL
        download_dir: Directory for downloaded files (fallback only)
        use_bigquery: Use BigQuery as primary data source (default: True)
        bigquery_project: BigQuery project ID (default: "gdelt-bq")
        google_credentials_path: Path to Google Cloud credentials JSON
        target_countries: List of country codes to target (default: Horn of Africa)
        use_events_data: Use Events data vs GKG data (default: True - Events)
        fallback_enabled: Enable fallback to API/files if BigQuery fails
        **kwargs: Additional configuration options
        
    Returns:
        Configured GDELTCrawler instance with BigQuery priority
    """
    # Set intelligent defaults
    config = GDELTCrawlerConfig(
        database_url=database_url,
        download_dir=Path(download_dir) if download_dir else None,
        use_bigquery=use_bigquery,
        bigquery_project=bigquery_project,
        google_credentials_path=google_credentials_path,
        target_countries=target_countries or ["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
        use_events_data=use_events_data,
        fallback_enabled=fallback_enabled,
        **kwargs
    )
    
    logger.info(f"🚀 Creating GDELT crawler with BigQuery {'enabled' if use_bigquery else 'disabled'}")
    logger.info(f"   Primary data source: {'BigQuery Events' if use_events_data else 'BigQuery GKG'}")
    logger.info(f"   Fallback enabled: {fallback_enabled}")
    logger.info(f"   Target countries: {config.target_countries}")
    
    return GDELTCrawler(config)


def create_bigquery_gdelt_crawler(
    database_url: str,
    bigquery_project: str = "gdelt-bq",
    google_credentials_path: Optional[str] = None,
    target_countries: Optional[List[str]] = None,
    use_events_data: bool = True,
    **kwargs
) -> GDELTCrawler:
    """
    Create a GDELT crawler configured specifically for BigQuery access.
    
    Args:
        database_url: PostgreSQL database URL (required)
        bigquery_project: BigQuery project ID
        google_credentials_path: Path to Google Cloud credentials JSON
        target_countries: List of country codes to target
        use_events_data: Use Events data vs GKG data (Events recommended)
        **kwargs: Additional configuration options
        
    Returns:
        GDELT crawler configured for BigQuery-only operation
    """
    return create_gdelt_crawler(
        database_url=database_url,
        use_bigquery=True,
        bigquery_project=bigquery_project,
        google_credentials_path=google_credentials_path,
        target_countries=target_countries,
        use_events_data=use_events_data,
        fallback_enabled=False,  # BigQuery-only, no fallbacks
        **kwargs
    )


def create_legacy_gdelt_crawler(
    database_url: Optional[str] = None,
    download_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> GDELTCrawler:
    """
    Create a legacy GDELT crawler using file downloads (not recommended).
    
    Args:
        database_url: PostgreSQL database URL
        download_dir: Directory for downloaded files
        **kwargs: Additional configuration options
        
    Returns:
        GDELT crawler configured for file-based operation
    """
    logger.warning("⚠️ Creating legacy file-based GDELT crawler. BigQuery method is recommended for better performance.")
    
    return create_gdelt_crawler(
        database_url=database_url,
        download_dir=download_dir,
        use_bigquery=False,  # Disable BigQuery
        fallback_enabled=False,  # Files-only
        **kwargs
    )


def create_gdelt_client(**kwargs) -> 'GDELTClient':
    """
    Create a basic GDELT API client.
    
    Args:
        **kwargs: Configuration options for the client
        
    Returns:
        GDELTClient instance
    """
    if not _GDELT_API_AVAILABLE:
        raise ImportError("GDELT API client not available")
    
    return GDELTClient(**kwargs)


# Convenience functions  
async def query_gdelt_events(
    query: str,
    days_back: int = 7,
    max_records: int = 250,
    database_url: Optional[str] = None,
    use_bigquery: bool = True,
    **kwargs
) -> List[GDELTEvent]:
    """
    Quick function to query GDELT events using BigQuery by default.
    
    Args:
        query: Search query
        days_back: Number of days to search back
        max_records: Maximum records to return
        database_url: Database URL for storing results
        use_bigquery: Use BigQuery method (default: True)
        **kwargs: Additional query parameters
        
    Returns:
        List of GDELT events
    """
    crawler = create_gdelt_crawler(
        database_url=database_url,
        use_bigquery=use_bigquery,
        **kwargs
    )
    try:
        await crawler.start()
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        return await crawler.query_events(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_records=max_records,
            **kwargs
        )
    finally:
        await crawler.stop()


async def run_bigquery_etl(
    database_url: str,
    start_date: datetime,
    end_date: datetime,
    target_countries: Optional[List[str]] = None,
    use_events_data: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to run BigQuery ETL for a date range.
    
    Args:
        database_url: PostgreSQL database URL (required)
        start_date: Start date for processing
        end_date: End date for processing  
        target_countries: List of country codes to target
        use_events_data: Use Events vs GKG data (Events recommended)
        **kwargs: Additional configuration options
        
    Returns:
        ETL processing summary with metrics
    """
    crawler = create_bigquery_gdelt_crawler(
        database_url=database_url,
        target_countries=target_countries,
        use_events_data=use_events_data,
        **kwargs
    )
    
    try:
        await crawler.start()
        
        if use_events_data:
            return await crawler.process_events_data(start_date, end_date)
        else:
            return await crawler.process_bigquery_data(start_date, end_date)
    finally:
        await crawler.stop()


async def download_gdelt_daily(
    date: Optional[datetime] = None,
    datasets: Optional[List[str]] = None,
    download_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Quick function to download GDELT daily files.
    
    Args:
        date: Date to download (defaults to yesterday)
        datasets: Datasets to download
        download_dir: Download directory
        
    Returns:
        Dictionary of downloaded file paths
    """
    crawler = create_gdelt_crawler(download_dir=download_dir)
    try:
        await crawler.start()
        
        if date is None:
            date = datetime.now(timezone.utc) - timedelta(days=1)
        
        return await crawler.download_daily_files(date, datasets)
    finally:
        await crawler.stop()


# Export all public components
__all__ = [
    # Main classes
    "GDELTCrawler",
    "GDELTCrawlerConfig",
    "GDELTEvent",
    
    # Factory functions
    "create_gdelt_crawler",
    "create_bigquery_gdelt_crawler", 
    "create_legacy_gdelt_crawler",
    "create_gdelt_client",
    
    # Convenience functions
    "query_gdelt_events",
    "run_bigquery_etl",
    "download_gdelt_daily",
    
    # Version info
    "__version__",
    "__author__",
    "__company__",
    "__website__",
    "__license__"
]

# Conditional exports based on availability
if _GDELT_API_AVAILABLE:
    __all__.extend([
        "GDELTClient",
        "GDELTQueryParameters", 
        "GDELTArticle",
        "GDELTDateRange",
        "GDELTMode",
        "GDELTFormat"
    ])

if _GDELT_BULK_AVAILABLE:
    __all__.extend([
        "GDELTFileDownloader",
        "GDELTDataset",
        "DownloadConfig",
        "GDELTScheduler",
        "GDELTScheduleConfig"
    ])

if _GDELT_DATABASE_AVAILABLE:
    __all__.extend([
        "GDELTDatabaseETL",
        "GDELTBigQueryETL",
        "BigQueryETLConfig",
        "create_bigquery_etl",
        "GDELTEventsETL",
        "EventsETLConfig",
        "create_events_etl",
        "InformationUnit"
    ])

if _GDELT_MONITORING_AVAILABLE:
    __all__.extend([
        "GDELTAlertSystem",
        "GDELTMetrics"
    ])

# Backward compatibility aliases
GDELTCrawler = GDELTCrawler

# Add alias to exports
__all__.append("GDELTCrawler")

# Package initialization
logger.info(f"Enhanced GDELT Crawler Package v{__version__} initialized")
logger.info(f"🚀 BigQuery-First Architecture: BigQuery → API → File Downloads")
logger.info(f"Author: {__author__} | Company: {__company__} ({__website__})")
logger.info(f"Available components: API={_GDELT_API_AVAILABLE}, "
           f"Bulk={_GDELT_BULK_AVAILABLE}, Database={_GDELT_DATABASE_AVAILABLE}, "
           f"Monitoring={_GDELT_MONITORING_AVAILABLE}")
logger.info(f"💡 Use create_gdelt_crawler() for BigQuery with intelligent fallback")
logger.info(f"💡 Use create_bigquery_gdelt_crawler() for BigQuery-only operation")