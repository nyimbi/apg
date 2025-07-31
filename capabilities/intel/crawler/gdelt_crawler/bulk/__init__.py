"""
GDELT Bulk File Processing Components
====================================

Components for automated daily file downloads, processing, and scheduling
of GDELT bulk data files including Events, Mentions, and GKG datasets.

Components:
- **GDELTFileDownloader**: Automated daily file downloads with resumption
- **GDELTScheduler**: Production scheduler for automated downloads
- **GDELTFileProcessor**: Advanced CSV parsing and data transformation

Features:
- Parallel downloads with bandwidth management
- Automatic decompression and validation
- Smart scheduling with retry logic
- Database-ready data transformation

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

try:
    from .file_downloader import (
        GDELTFileDownloader,
        GDELTDataset,
        DownloadConfig,
        DownloadProgress,
        DownloadResult,
        CompressionType
    )
    
    from .scheduler import (
        GDELTScheduler,
        ScheduleConfig,
        ScheduleFrequency,
        ScheduleStatus,
        ScheduleJob,
        SchedulerMetrics,
        create_daily_scheduler
    )
    
    from .file_processor import (
        GDELTFileProcessor,
        GDELTFieldType,
        GDELTField,
        ProcessingStats,
        process_gdelt_file_simple
    )
    
    __all__ = [
        # File Downloader
        'GDELTFileDownloader',
        'GDELTDataset',
        'DownloadConfig', 
        'DownloadProgress',
        'DownloadResult',
        'CompressionType',
        
        # Scheduler
        'GDELTScheduler',
        'ScheduleConfig',
        'ScheduleFrequency',
        'ScheduleStatus', 
        'ScheduleJob',
        'SchedulerMetrics',
        'create_daily_scheduler',
        
        # File Processor
        'GDELTFileProcessor',
        'GDELTFieldType',
        'GDELTField',
        'ProcessingStats',
        'process_gdelt_file_simple'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    __all__ = []
    import logging
    logging.getLogger(__name__).warning(f"GDELT bulk components not available: {e}")

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"