"""
GDELT Scheduler Wrapper
=======================

A wrapper around the utils/scheduling package that provides GDELT-specific
scheduling functionality while leveraging the existing scheduling infrastructure.

This module replaces the custom scheduler implementation with proper integration
of the packages_enhanced/utils/scheduling system.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import asyncio

# Import from utils packages
from ....utils.scheduling import (
    Scheduler,
    JobManager,
    TaskExecutor,
    ScheduleConfig as BaseScheduleConfig,
    Job,
    JobStatus,
    CronParser,
    ResourceMonitor
)

from .file_downloader import GDELTFileDownloader, GDELTDataset

logger = logging.getLogger(__name__)


@dataclass
class GDELTScheduleConfig(BaseScheduleConfig):
    """GDELT-specific scheduling configuration."""
    # GDELT-specific scheduling fields
    schedule_type: str = 'daily'
    schedule_time: str = '02:00'
    datasets: List[Union[str, GDELTDataset]] = None
    days_back_on_start: int = 1
    enable_etl: bool = True
    etl_batch_size: int = 1000
    auto_cleanup_downloads: bool = False
    keep_downloads_days: int = 7
    alert_on_failure: bool = True
    max_consecutive_failures: int = 3
    # Legacy compatibility
    max_concurrent_jobs: Optional[int] = None
    enable_monitoring: bool = True
    
    def __post_init__(self):
        # Handle legacy max_concurrent_jobs -> max_concurrent_tasks mapping
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        if self.max_concurrent_jobs is not None and not hasattr(self, '_max_concurrent_tasks_set'):
            self.max_concurrent_tasks = self.max_concurrent_jobs
        
        # Initialize datasets if not provided
        if self.datasets is None:
            self.datasets = [GDELTDataset.EVENTS, GDELTDataset.MENTIONS, GDELTDataset.GKG]
        
        # Convert string datasets to enums
        validated_datasets = []
        for dataset in self.datasets:
            if isinstance(dataset, str):
                validated_datasets.append(GDELTDataset(dataset.lower()))
            else:
                validated_datasets.append(dataset)
        self.datasets = validated_datasets


class GDELTSchedulerWrapper:
    """
    Wrapper for GDELT-specific scheduling using utils/scheduling.
    
    This replaces the custom scheduler implementation with proper integration
    of the existing scheduling infrastructure.
    """
    
    def __init__(
        self,
        file_downloader: GDELTFileDownloader,
        config: GDELTScheduleConfig,
        database_etl: Optional[Any] = None,
        metrics_wrapper: Optional[Any] = None
    ):
        """
        Initialize GDELT scheduler wrapper.
        
        Args:
            file_downloader: GDELT file downloader instance
            config: GDELT scheduling configuration
            database_etl: Optional database ETL instance
            metrics_wrapper: Optional metrics wrapper instance
        """
        self.file_downloader = file_downloader
        self.config = config
        self.database_etl = database_etl
        self.metrics_wrapper = metrics_wrapper
        
        # Initialize scheduler from utils
        base_config = BaseScheduleConfig(
            max_concurrent_tasks=getattr(config, 'max_concurrent_tasks', getattr(config, 'max_concurrent_jobs', 10)),
            retry_attempts=getattr(config, 'retry_attempts', getattr(config, 'max_retries', 3)),
            retry_delay=getattr(config, 'retry_delay', 60),
            enable_persistence=getattr(config, 'enable_persistence', True),
            enable_metrics=getattr(config, 'enable_monitoring', True)
        )
        
        self.scheduler = Scheduler(base_config)
        self.job_manager = JobManager()
        self.task_executor = TaskExecutor(max_workers=getattr(config, 'max_concurrent_tasks', getattr(config, 'max_concurrent_jobs', 10)))
        self.resource_monitor = ResourceMonitor()
        
        # GDELT-specific state
        self._consecutive_failures = 0
        self._monitoring_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Register GDELT job handlers
        self._register_job_handlers()
        
        logger.info("GDELT Scheduler initialized with utils/scheduling")
    
    async def start(self):
        """Start the scheduler."""
        # Start utils scheduler
        await self.scheduler.start()
        
        # Check for missing files on startup
        if self.config.days_back_on_start > 0:
            await self._check_missing_files()
        
        # Schedule regular GDELT downloads
        await self._schedule_gdelt_downloads()
        
        logger.info("GDELT Scheduler started")
    
    async def stop(self):
        """Stop the scheduler gracefully."""
        await self.scheduler.stop()
        # Note: TaskExecutor doesn't have a shutdown method
        # Running tasks will finish naturally
        logger.info("GDELT Scheduler stopped")
    
    async def pause(self):
        """Pause the scheduler."""
        await self.scheduler.pause()
        logger.info("GDELT Scheduler paused")
    
    async def resume(self):
        """Resume the scheduler."""
        await self.scheduler.resume()
        logger.info("GDELT Scheduler resumed")
    
    def _register_job_handlers(self):
        """Register GDELT-specific job handlers."""
        # Register download handler
        self.scheduler.register_handler(
            'gdelt_download',
            self._handle_download_job
        )
        
        # Register ETL handler
        self.scheduler.register_handler(
            'gdelt_etl',
            self._handle_etl_job
        )
        
        # Register cleanup handler
        self.scheduler.register_handler(
            'gdelt_cleanup',
            self._handle_cleanup_job
        )
    
    async def _schedule_gdelt_downloads(self):
        """Schedule regular GDELT downloads based on configuration."""
        # Daily download job
        if self.config.schedule_type == 'daily':
            # Use cron expression for daily schedule
            cron_expr = f"0 {self.config.schedule_time.split(':')[1]} {self.config.schedule_time.split(':')[0]} * * *"
            
            await self.scheduler.schedule_recurring(
                job_type='gdelt_download',
                cron_expression=cron_expr,
                job_data={
                    'datasets': [d.value for d in self.config.datasets],
                    'date_offset': -1  # Download yesterday's data
                },
                job_id='gdelt_daily_download'
            )
        
        # Schedule cleanup if enabled
        if self.config.auto_cleanup_downloads:
            # Run cleanup daily at 4 AM
            await self.scheduler.schedule_recurring(
                job_type='gdelt_cleanup',
                cron_expression="0 0 4 * * *",
                job_data={
                    'keep_days': self.config.keep_downloads_days
                },
                job_id='gdelt_cleanup'
            )
    
    async def schedule_immediate_download(
        self,
        date: datetime,
        datasets: Optional[List[Union[str, GDELTDataset]]] = None
    ) -> str:
        """
        Schedule an immediate download for a specific date.
        
        Args:
            date: Date to download files for
            datasets: List of datasets to download
            
        Returns:
            Job ID for tracking
        """
        if datasets is None:
            datasets = self.config.datasets
        
        # Convert to dataset values
        dataset_values = []
        for d in datasets:
            if isinstance(d, str):
                dataset_values.append(d)
            else:
                dataset_values.append(d.value)
        
        # Create job
        job = await self.scheduler.schedule_once(
            job_type='gdelt_download',
            job_data={
                'date': date.isoformat(),
                'datasets': dataset_values,
                'immediate': True
            },
            run_at=datetime.utcnow()  # Run immediately
        )
        
        logger.info(f"Scheduled immediate download for {date.date()}: {job.job_id}")
        return job.job_id
    
    async def schedule_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        datasets: Optional[List[Union[str, GDELTDataset]]] = None
    ) -> List[str]:
        """
        Schedule downloads for a range of dates.
        
        Args:
            start_date: Start date for downloads
            end_date: End date for downloads
            datasets: List of datasets to download
            
        Returns:
            List of job IDs
        """
        job_ids = []
        current_date = start_date
        
        while current_date <= end_date:
            job_id = await self.schedule_immediate_download(current_date, datasets)
            job_ids.append(job_id)
            current_date += timedelta(days=1)
        
        logger.info(f"Scheduled {len(job_ids)} downloads from {start_date.date()} to {end_date.date()}")
        return job_ids
    
    async def _handle_download_job(self, job: Job) -> Dict[str, Any]:
        """Handle GDELT download job."""
        job_data = job.data
        
        # Extract job parameters
        if 'date' in job_data:
            date = datetime.fromisoformat(job_data['date'])
        else:
            # Use date offset (for recurring jobs)
            date_offset = job_data.get('date_offset', -1)
            date = datetime.utcnow() + timedelta(days=date_offset)
        
        datasets = job_data.get('datasets', [d.value for d in self.config.datasets])
        
        # Perform downloads
        downloaded_files = {}
        errors = []
        
        for dataset in datasets:
            try:
                # Convert string to enum if needed
                if isinstance(dataset, str):
                    dataset_enum = GDELTDataset(dataset)
                else:
                    dataset_enum = dataset
                
                # Download file
                file_path = await self.file_downloader.download_daily_file(
                    date=date,
                    dataset=dataset_enum
                )
                
                downloaded_files[dataset] = str(file_path)
                
                # Record metrics if available
                if self.metrics_wrapper:
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    self.metrics_wrapper.record_download_metrics(
                        dataset=dataset,
                        file_size_bytes=file_size,
                        download_time=0  # Would need to track actual time
                    )
                
                logger.info(f"Downloaded {dataset} for {date.date()}: {file_path}")
                
            except Exception as e:
                error_msg = f"Failed to download {dataset} for {date.date()}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                if self.metrics_wrapper:
                    self.metrics_wrapper.record_download_failure(dataset, str(e))
        
        # Process with ETL if enabled and files were downloaded
        processed_counts = {}
        if self.config.enable_etl and self.database_etl and downloaded_files:
            etl_job = await self.scheduler.schedule_once(
                job_type='gdelt_etl',
                job_data={
                    'date': date.isoformat(),
                    'downloaded_files': downloaded_files,
                    'batch_size': self.config.etl_batch_size
                },
                run_at=datetime.utcnow()
            )
            
            # Wait for ETL to complete (with timeout)
            try:
                etl_result = await asyncio.wait_for(
                    self._wait_for_job_completion(etl_job.job_id),
                    timeout=3600  # 1 hour timeout
                )
                if etl_result and 'processed_counts' in etl_result:
                    processed_counts = etl_result['processed_counts']
            except asyncio.TimeoutError:
                logger.error(f"ETL timeout for job {etl_job.job_id}")
        
        # Update consecutive failures
        if errors:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                await self._trigger_failure_alert(errors)
        else:
            self._consecutive_failures = 0
        
        # Trigger monitoring callbacks
        await self._trigger_monitoring_callbacks({
            'job_id': job.job_id,
            'date': date.isoformat(),
            'downloaded_files': downloaded_files,
            'processed_counts': processed_counts,
            'errors': errors
        })
        
        return {
            'downloaded_files': downloaded_files,
            'processed_counts': processed_counts,
            'errors': errors,
            'success': len(errors) == 0
        }
    
    async def _handle_etl_job(self, job: Job) -> Dict[str, Any]:
        """Handle GDELT ETL job."""
        job_data = job.data
        downloaded_files = job_data.get('downloaded_files', {})
        batch_size = job_data.get('batch_size', self.config.etl_batch_size)
        date = datetime.fromisoformat(job_data.get('date'))
        
        processed_counts = {}
        errors = []
        
        for dataset, file_path in downloaded_files.items():
            try:
                # Process file with ETL
                count = await self.database_etl.process_file(
                    file_path=Path(file_path),
                    dataset=dataset,
                    batch_size=batch_size
                )
                
                processed_counts[dataset] = count
                
                logger.info(f"Processed {count} records from {dataset}")
                
            except Exception as e:
                error_msg = f"Failed to process {dataset}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Record metrics
        if self.metrics_wrapper:
            if errors:
                self.metrics_wrapper.record_etl_failure(date, '; '.join(errors))
            else:
                self.metrics_wrapper.record_etl_completion(date, processed_counts)
        
        return {
            'processed_counts': processed_counts,
            'errors': errors,
            'success': len(errors) == 0
        }
    
    async def _handle_cleanup_job(self, job: Job) -> Dict[str, Any]:
        """Handle cleanup job to remove old downloads."""
        keep_days = job.data.get('keep_days', self.config.keep_downloads_days)
        cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
        
        cleaned_files = []
        errors = []
        
        try:
            download_dir = self.file_downloader.config.download_dir
            
            for file_path in download_dir.rglob('*.csv*'):
                # Check file age
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                    except Exception as e:
                        errors.append(f"Failed to delete {file_path}: {e}")
            
            logger.info(f"Cleaned up {len(cleaned_files)} old files")
            
        except Exception as e:
            error_msg = f"Cleanup job failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        return {
            'cleaned_files': cleaned_files,
            'errors': errors,
            'success': len(errors) == 0
        }
    
    async def _check_missing_files(self):
        """Check for and download missing files on startup."""
        end_date = datetime.utcnow() - timedelta(days=1)
        start_date = end_date - timedelta(days=self.config.days_back_on_start)
        
        logger.info(f"Checking for missing files from {start_date.date()} to {end_date.date()}")
        
        missing_jobs = []
        current_date = start_date
        
        while current_date <= end_date:
            # Check if files exist for this date
            needs_download = False
            for dataset in self.config.datasets:
                local_path = self.file_downloader._build_local_path(current_date, dataset)
                if not local_path.exists():
                    needs_download = True
                    break
            
            if needs_download:
                job_id = await self.schedule_immediate_download(current_date, self.config.datasets)
                missing_jobs.append(job_id)
            
            current_date += timedelta(days=1)
        
        if missing_jobs:
            logger.info(f"Scheduled {len(missing_jobs)} downloads for missing files")
        else:
            logger.info("No missing files found")
    
    async def _wait_for_job_completion(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Wait for a job to complete and return its result."""
        max_wait = 3600  # 1 hour
        check_interval = 5  # 5 seconds
        elapsed = 0
        
        while elapsed < max_wait:
            job = await self.job_manager.get_job(job_id)
            if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return job.result
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        return None
    
    async def _trigger_failure_alert(self, errors: List[str]):
        """Trigger alert for consecutive failures."""
        if not self.config.alert_on_failure:
            return
        
        alert_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'consecutive_failures': self._consecutive_failures,
            'recent_errors': errors[-5:],  # Last 5 errors
            'message': f"GDELT Scheduler has failed {self._consecutive_failures} times consecutively"
        }
        
        logger.error(f"ALERT: {alert_data['message']}")
        
        # Could integrate with alert manager here
        # if self.alert_manager:
        #     await self.alert_manager.send_alert('gdelt_scheduler_failure', alert_data)
    
    async def _trigger_monitoring_callbacks(self, event_data: Dict[str, Any]):
        """Trigger monitoring callbacks."""
        for callback in self._monitoring_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.error(f"Monitoring callback error: {e}")
    
    def add_monitoring_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a monitoring callback function."""
        self._monitoring_callbacks.append(callback)
    
    def remove_monitoring_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a monitoring callback function."""
        if callback in self._monitoring_callbacks:
            self._monitoring_callbacks.remove(callback)
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        job = await self.job_manager.get_job(job_id)
        if not job:
            return None
        
        return {
            'job_id': job.job_id,
            'type': job.job_type,
            'status': job.status.value,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'result': job.result,
            'error': job.error,
            'retry_count': job.retry_count
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics."""
        scheduler_stats = await self.scheduler.get_stats()
        resource_stats = await self.resource_monitor.get_current_usage()
        
        return {
            'scheduler': scheduler_stats,
            'resources': resource_stats,
            'gdelt_specific': {
                'consecutive_failures': self._consecutive_failures,
                'datasets': [d.value for d in self.config.datasets],
                'enable_etl': self.config.enable_etl,
                'auto_cleanup': self.config.auto_cleanup_downloads
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check scheduler health
        scheduler_health = await self.scheduler.health_check()
        health['components']['scheduler'] = scheduler_health
        
        # Check file downloader
        if hasattr(self.file_downloader, 'health_check'):
            downloader_health = await self.file_downloader.health_check()
            health['components']['file_downloader'] = downloader_health
        
        # Check database ETL
        if self.database_etl and hasattr(self.database_etl, 'health_check'):
            etl_health = await self.database_etl.health_check()
            health['components']['database_etl'] = etl_health
        
        # Determine overall status
        if any(comp.get('status') == 'error' for comp in health['components'].values()):
            health['overall_status'] = 'error'
        elif any(comp.get('status') == 'degraded' for comp in health['components'].values()):
            health['overall_status'] = 'degraded'
        
        return health


# Factory functions
def create_daily_scheduler(
    downloader: GDELTFileDownloader,
    schedule_time: str = "02:00",
    datasets: Optional[List[str]] = None,
    **kwargs
) -> GDELTSchedulerWrapper:
    """
    Create a daily GDELT scheduler with default configuration.
    
    Args:
        downloader: GDELT file downloader instance
        schedule_time: Time to run daily downloads (HH:MM format)
        datasets: List of datasets to download
        **kwargs: Additional configuration options
        
    Returns:
        Configured GDELTSchedulerWrapper instance
    """
    config = GDELTScheduleConfig(
        schedule_type='daily',
        schedule_time=schedule_time,
        datasets=datasets or ['events', 'mentions', 'gkg'],
        **kwargs
    )
    return GDELTSchedulerWrapper(downloader, config)


# Export components
__all__ = [
    'GDELTSchedulerWrapper',
    'GDELTScheduleConfig',
    'create_daily_scheduler'
]