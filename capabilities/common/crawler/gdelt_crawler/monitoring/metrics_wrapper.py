"""
GDELT Metrics Wrapper
=====================

A wrapper around the utils/monitoring and utils/performance packages that provides
GDELT-specific metrics collection while leveraging the existing monitoring infrastructure.

This module replaces the custom metrics implementation with proper integration
of the packages_enhanced/utils/monitoring and utils/performance systems.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import from utils packages
from ....utils.monitoring import MetricsCollector, PerformanceMonitor
from ....utils.performance import (
    SystemMonitor,
    AlertManager,
    AlertConfig,
    BenchmarkRunner,
    MemoryProfiler
)

logger = logging.getLogger(__name__)


class GDELTMetricsWrapper:
    """
    Wrapper for GDELT-specific metrics using utils/monitoring and utils/performance.
    
    This replaces the custom metrics implementation with proper integration
    of the existing monitoring infrastructure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GDELT metrics wrapper.
        
        Args:
            config: Configuration for metrics collection
        """
        config = config or {}
        
        # Initialize monitoring components from utils
        self.metrics_collector = MetricsCollector(
            enable_prometheus=False,
            buffer_size=10000
        )
        
        self.performance_monitor = PerformanceMonitor(
            config={
                'enable_profiling': config.get('enable_profiling', False),
                'enable_memory_tracking': config.get('enable_memory_tracking', True)
            }
        )
        
        self.system_monitor = SystemMonitor(
            config={
                'monitor_interval': config.get('system_monitor_interval', 60),
                'include_network': True,
                'include_disk': True
            }
        )
        
        self.alert_manager = AlertManager(
            config=AlertConfig(
                output_dir="/tmp/lindela_alerts",
                enable_alert_grouping=True,
                alert_grouping_window=300,
                max_alerts_per_rule=10
            )
        )
        
        # GDELT-specific metric names
        self.metric_names = {
            # ETL metrics
            'etl_total_processed': 'gdelt.etl.records.processed',
            'etl_completion_rate': 'gdelt.etl.completion.rate',
            'etl_failure_rate': 'gdelt.etl.failure.rate',
            'etl_duration': 'gdelt.etl.duration.seconds',
            
            # Data quality metrics
            'quality_validation_rate': 'gdelt.quality.validation.rate',
            'quality_error_rate': 'gdelt.quality.error.rate',
            
            # Download metrics
            'download_size': 'gdelt.download.size.bytes',
            'download_duration': 'gdelt.download.duration.seconds',
            'download_speed': 'gdelt.download.speed.mbps',
            
            # Conflict detection metrics
            'conflicts_detected': 'gdelt.conflicts.detected.count',
            'conflicts_severity': 'gdelt.conflicts.severity.score',
            'conflicts_fatalities': 'gdelt.conflicts.fatalities.total',
            
            # ML processing metrics
            'ml_processed': 'gdelt.ml.processed.count',
            'ml_confidence': 'gdelt.ml.confidence.score',
            'ml_duration': 'gdelt.ml.duration.ms'
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.system_monitor.start()
        if hasattr(self.alert_manager, 'start'):
            self.alert_manager.start()
        logger.info("GDELT metrics monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.system_monitor.stop()
        if hasattr(self.alert_manager, 'stop'):
            self.alert_manager.stop()
        logger.info("GDELT metrics monitoring stopped")
    
    # ==================== ETL METRICS ====================
    
    def record_etl_completion(self, date: datetime, processed_counts: Dict[str, int]):
        """Record ETL completion metrics."""
        total_processed = sum(processed_counts.values())
        
        # Record metrics using utils
        self.metrics_collector.record_counter(
            self.metric_names['etl_total_processed'],
            total_processed,
            tags={'date': date.isoformat()}
        )
        
        self.metrics_collector.record_gauge(
            self.metric_names['etl_completion_rate'],
            1.0,  # Successful completion
            tags={'status': 'success'}
        )
        
        # Record per-dataset metrics
        for dataset, count in processed_counts.items():
            self.metrics_collector.record_counter(
                f'gdelt.etl.{dataset}.processed',
                count,
                tags={'dataset': dataset, 'date': date.isoformat()}
            )
        
        # Track performance
        with self.performance_monitor.track_operation('etl_completion'):
            logger.info(f"ETL metrics recorded: {total_processed} records processed")
    
    def record_etl_failure(self, date: datetime, error: str):
        """Record ETL failure metrics."""
        self.metrics_collector.record_counter(
            self.metric_names['etl_failure_rate'],
            1,
            tags={'date': date.isoformat(), 'error_type': self._classify_error(error)}
        )
        
        # Trigger alert if needed
        self.alert_manager.check_alert(
            'etl_failure',
            {'error': error, 'date': date.isoformat()}
        )
        
        logger.warning(f"ETL failure recorded for {date.date()}: {error}")
    
    def record_processing_time(self, operation: str, duration_seconds: float):
        """Record processing time metrics."""
        self.metrics_collector.record_histogram(
            self.metric_names['etl_duration'],
            duration_seconds,
            tags={'operation': operation}
        )
        
        # Track with performance monitor
        self.performance_monitor.record_timing(
            f'gdelt.processing.{operation}',
            duration_seconds * 1000  # Convert to milliseconds
        )
    
    # ==================== DATA QUALITY METRICS ====================
    
    def record_data_quality(self, dataset: str, stats: Dict[str, Any]):
        """Record data quality metrics."""
        total_records = stats.get('total_records', 0)
        valid_records = stats.get('valid_records', 0)
        invalid_records = stats.get('invalid_records', 0)
        
        if total_records > 0:
            validation_rate = valid_records / total_records
            error_rate = invalid_records / total_records
            
            self.metrics_collector.record_gauge(
                self.metric_names['quality_validation_rate'],
                validation_rate,
                tags={'dataset': dataset}
            )
            
            self.metrics_collector.record_gauge(
                self.metric_names['quality_error_rate'],
                error_rate,
                tags={'dataset': dataset}
            )
        
        # Record counts
        self.metrics_collector.record_counter(
            f'gdelt.quality.{dataset}.total',
            total_records
        )
        self.metrics_collector.record_counter(
            f'gdelt.quality.{dataset}.valid',
            valid_records
        )
        self.metrics_collector.record_counter(
            f'gdelt.quality.{dataset}.invalid',
            invalid_records
        )
    
    def record_ml_processing(self, processed_count: int, high_confidence_count: int):
        """Record ML processing metrics."""
        self.metrics_collector.record_counter(
            self.metric_names['ml_processed'],
            processed_count
        )
        
        if processed_count > 0:
            confidence_rate = high_confidence_count / processed_count
            self.metrics_collector.record_gauge(
                self.metric_names['ml_confidence'],
                confidence_rate,
                tags={'threshold': 'high'}
            )
    
    # ==================== CONFLICT METRICS ====================
    
    def record_conflict_detection(self, total_events: int, conflict_events: int, high_severity: int):
        """Record conflict detection metrics."""
        self.metrics_collector.record_counter(
            self.metric_names['conflicts_detected'],
            conflict_events
        )
        
        if total_events > 0:
            conflict_rate = conflict_events / total_events
            severity_rate = high_severity / total_events
            
            self.metrics_collector.record_gauge(
                'gdelt.conflicts.rate',
                conflict_rate
            )
            
            self.metrics_collector.record_gauge(
                'gdelt.conflicts.severity.rate',
                severity_rate
            )
        
        # Check for alerts
        if conflict_rate > 0.3:  # More than 30% conflict events
            self.alert_manager.trigger_alert(
                'high_conflict_rate',
                {
                    'rate': conflict_rate,
                    'total_events': total_events,
                    'conflict_events': conflict_events
                }
            )
    
    def record_fatality_statistics(self, total_fatalities: int, events_with_fatalities: int):
        """Record fatality statistics."""
        self.metrics_collector.record_counter(
            self.metric_names['conflicts_fatalities'],
            total_fatalities
        )
        
        self.metrics_collector.record_counter(
            'gdelt.conflicts.fatal_events',
            events_with_fatalities
        )
        
        if events_with_fatalities > 0:
            avg_fatalities = total_fatalities / events_with_fatalities
            self.metrics_collector.record_gauge(
                'gdelt.conflicts.fatalities.average',
                avg_fatalities
            )
    
    # ==================== DOWNLOAD METRICS ====================
    
    def record_download_metrics(self, dataset: str, file_size_bytes: int, download_time: float):
        """Record file download metrics."""
        self.metrics_collector.record_histogram(
            self.metric_names['download_size'],
            file_size_bytes,
            tags={'dataset': dataset}
        )
        
        self.metrics_collector.record_histogram(
            self.metric_names['download_duration'],
            download_time,
            tags={'dataset': dataset}
        )
        
        if download_time > 0:
            speed_mbps = (file_size_bytes / (1024**2)) / download_time * 8
            self.metrics_collector.record_gauge(
                self.metric_names['download_speed'],
                speed_mbps,
                tags={'dataset': dataset}
            )
    
    def record_download_failure(self, dataset: str, error: str):
        """Record download failure metrics."""
        self.metrics_collector.record_counter(
            'gdelt.download.failures',
            1,
            tags={
                'dataset': dataset,
                'error_type': self._classify_error(error)
            }
        )
    
    # ==================== UTILITIES ====================
    
    def _classify_error(self, error: str) -> str:
        """Classify error into categories."""
        error_lower = error.lower()
        
        if any(term in error_lower for term in ['network', 'connection', 'timeout', 'dns']):
            return "network"
        elif any(term in error_lower for term in ['permission', 'access', 'auth']):
            return "permission"
        elif any(term in error_lower for term in ['disk', 'space', 'storage']):
            return "storage"
        elif any(term in error_lower for term in ['memory', 'out of memory']):
            return "memory"
        elif any(term in error_lower for term in ['parse', 'format', 'invalid']):
            return "data_format"
        else:
            return "other"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        # Get metrics from collectors
        metrics_data = self.metrics_collector.get_all_metrics()
        performance_data = self.performance_monitor.get_summary()
        system_data = self.system_monitor.get_current_stats()
        
        # Build GDELT-specific summary
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'etl': {
                'total_processed': metrics_data.get(self.metric_names['etl_total_processed'], 0),
                'completion_rate': metrics_data.get(self.metric_names['etl_completion_rate'], 0),
                'failure_rate': metrics_data.get(self.metric_names['etl_failure_rate'], 0)
            },
            'data_quality': {
                'validation_rate': metrics_data.get(self.metric_names['quality_validation_rate'], 0),
                'error_rate': metrics_data.get(self.metric_names['quality_error_rate'], 0)
            },
            'conflicts': {
                'total_detected': metrics_data.get(self.metric_names['conflicts_detected'], 0),
                'total_fatalities': metrics_data.get(self.metric_names['conflicts_fatalities'], 0)
            },
            'ml_processing': {
                'total_processed': metrics_data.get(self.metric_names['ml_processed'], 0),
                'avg_confidence': metrics_data.get(self.metric_names['ml_confidence'], 0)
            },
            'performance': performance_data,
            'system': system_data
        }
        
        return summary
    
    def get_metric_history(self, metric_name: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for visualization."""
        # Map GDELT metric name to internal name
        internal_name = self.metric_names.get(metric_name, metric_name)
        
        # Get history from metrics collector
        return self.metrics_collector.get_metric_history(
            internal_name,
            time_window_hours=hours_back
        )
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format."""
        return self.metrics_collector.export(format_type)
    
    @property
    def benchmark_runner(self) -> BenchmarkRunner:
        """Get benchmark runner for performance testing."""
        return BenchmarkRunner()
    
    @property
    def memory_profiler(self) -> MemoryProfiler:
        """Get memory profiler for memory analysis."""
        return MemoryProfiler()


# Factory function
def create_gdelt_metrics(config: Optional[Dict[str, Any]] = None) -> GDELTMetricsWrapper:
    """Create GDELT metrics wrapper with configuration."""
    return GDELTMetricsWrapper(config)


# Export components
__all__ = [
    'GDELTMetricsWrapper',
    'create_gdelt_metrics'
]