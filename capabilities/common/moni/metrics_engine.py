#!/usr/bin/env python3
"""
APG Monitoring - Metrics Collection Engine
Real-time metrics ingestion with sub-second latency and high-cardinality support

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from uuid6 import uuid7
def uuid7str() -> str: return str(uuid7())

from .models import MonitoringMetric, MetricType, DataRetentionPolicy


class MetricFormat(str, Enum):
	"""Supported metric formats"""
	PROMETHEUS = "prometheus"
	OPENTELEMETRY = "opentelemetry" 
	STATSD = "statsd"
	CUSTOM = "custom"
	JSON = "json"


class BatchProcessingMode(str, Enum):
	"""Batch processing modes"""
	TIME_BASED = "time_based"
	SIZE_BASED = "size_based"
	HYBRID = "hybrid"


@dataclass
class MetricsBatch:
	"""Metrics batch for processing"""
	batch_id: str
	metrics: List[MonitoringMetric]
	format_type: MetricFormat
	created_at: datetime
	tenant_id: str
	source: str
	
	def size(self) -> int:
		return len(self.metrics)
	
	def age_seconds(self) -> float:
		return (datetime.utcnow() - self.created_at).total_seconds()


class MetricsBuffer:
	"""High-performance circular buffer for metrics"""
	
	def __init__(self, max_size: int = 100000):
		assert max_size > 0, "Buffer size must be positive"
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)
		self.lock = asyncio.Lock()
		self._metrics_by_tenant: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
		
	async def add_metric(self, metric: MonitoringMetric) -> None:
		"""Add metric to buffer with tenant isolation"""
		async with self.lock:
			self.buffer.append(metric)
			self._metrics_by_tenant[metric.tenant_id].append(metric)
	
	async def add_batch(self, metrics: List[MonitoringMetric]) -> None:
		"""Add metrics batch efficiently"""
		async with self.lock:
			for metric in metrics:
				self.buffer.append(metric)
				self._metrics_by_tenant[metric.tenant_id].append(metric)
	
	async def get_recent_metrics(self, limit: int = 1000, tenant_id: str = None) -> List[MonitoringMetric]:
		"""Get recent metrics with optional tenant filtering"""
		async with self.lock:
			if tenant_id:
				tenant_metrics = list(self._metrics_by_tenant[tenant_id])
				return tenant_metrics[-limit:] if len(tenant_metrics) > limit else tenant_metrics
			else:
				buffer_list = list(self.buffer)
				return buffer_list[-limit:] if len(buffer_list) > limit else buffer_list
	
	def get_stats(self) -> dict:
		"""Get buffer statistics"""
		return {
			'total_metrics': len(self.buffer),
			'buffer_utilization': len(self.buffer) / self.max_size,
			'tenant_count': len(self._metrics_by_tenant),
			'tenant_distribution': {
				tenant: len(metrics) for tenant, metrics in self._metrics_by_tenant.items()
			}
		}


class MetricsCollectionEngine:
	"""
	High-performance metrics collection engine with intelligent batching
	Supports multiple formats and provides sub-second ingestion latency
	"""
	
	def __init__(self, config: dict = None):
		self.config = config or {}
		self.running = False
		
		# Performance configuration
		self.batch_size = self.config.get('batch_size', 1000)
		self.batch_timeout_seconds = self.config.get('batch_timeout_seconds', 5)
		self.max_ingestion_rate = self.config.get('max_ingestion_rate', 100000)  # metrics/sec
		self.buffer_size = self.config.get('buffer_size', 500000)
		
		# Processing components
		self.metrics_buffer = MetricsBuffer(self.buffer_size)
		self.processing_queues: Dict[str, asyncio.Queue] = {}
		self.format_parsers: Dict[MetricFormat, callable] = {}
		self.batch_processors: List[asyncio.Task] = []
		
		# Performance tracking
		self.ingestion_stats = {
			'total_ingested': 0,
			'ingestion_rate': 0.0,
			'avg_latency_ms': 0.0,
			'last_reset': datetime.utcnow(),
			'format_distribution': defaultdict(int),
			'tenant_distribution': defaultdict(int)
		}
		
		# Rate limiting
		self.rate_limiter = asyncio.Semaphore(1000)
		self.tenant_rate_limiters: Dict[str, asyncio.Semaphore] = defaultdict(
			lambda: asyncio.Semaphore(100)
		)
		
		self._setup_format_parsers()
		self._log_info("Metrics collection engine initialized")
	
	def _log_info(self, message: str) -> None:
		"""Log information message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] [MetricsEngine] {message}")
	
	def _setup_format_parsers(self) -> None:
		"""Setup format parsers for different metric formats"""
		self.format_parsers = {
			MetricFormat.PROMETHEUS: self._parse_prometheus_format,
			MetricFormat.OPENTELEMETRY: self._parse_opentelemetry_format,
			MetricFormat.STATSD: self._parse_statsd_format,
			MetricFormat.JSON: self._parse_json_format,
			MetricFormat.CUSTOM: self._parse_custom_format
		}
	
	async def initialize(self) -> None:
		"""Initialize the metrics collection engine"""
		assert not self.running, "Engine is already running"
		
		# Create processing queues
		for tenant_id in ['default', 'system']:
			self.processing_queues[tenant_id] = asyncio.Queue(maxsize=10000)
		
		# Start batch processors
		for i in range(self.config.get('batch_processors', 4)):
			processor = asyncio.create_task(self._batch_processor_loop(f"processor_{i}"))
			self.batch_processors.append(processor)
		
		# Start performance monitor
		asyncio.create_task(self._performance_monitor_loop())
		
		self.running = True
		self._log_info("Metrics collection engine started successfully")
	
	async def shutdown(self) -> None:
		"""Gracefully shutdown the engine"""
		if not self.running:
			return
		
		self.running = False
		
		# Cancel batch processors
		for processor in self.batch_processors:
			processor.cancel()
		
		# Wait for processors to finish current work
		await asyncio.gather(*self.batch_processors, return_exceptions=True)
		
		self._log_info("Metrics collection engine shutdown complete")
	
	async def ingest_metric(self, metric_data: dict, format_type: MetricFormat = MetricFormat.JSON,
						   tenant_id: str = None, source: str = "unknown") -> bool:
		"""
		Ingest single metric with format parsing and validation
		Returns True if successfully ingested
		"""
		start_time = time.time()
		
		try:
			# Rate limiting
			tenant_id = tenant_id or "default"
			async with self.tenant_rate_limiters[tenant_id]:
				# Parse metric based on format
				parser = self.format_parsers.get(format_type)
				if not parser:
					self._log_info(f"Unsupported format: {format_type}")
					return False
				
				metric = await parser(metric_data, tenant_id, source)
				if not metric:
					return False
				
				# Add to buffer
				await self.metrics_buffer.add_metric(metric)
				
				# Update statistics
				await self._update_ingestion_stats(format_type, tenant_id, start_time)
				
				# Queue for batch processing
				if tenant_id not in self.processing_queues:
					self.processing_queues[tenant_id] = asyncio.Queue(maxsize=10000)
				
				try:
					self.processing_queues[tenant_id].put_nowait(metric)
				except asyncio.QueueFull:
					self._log_info(f"Processing queue full for tenant {tenant_id}")
					return False
				
				return True
				
		except Exception as e:
			self._log_info(f"Error ingesting metric: {e}")
			return False
	
	async def ingest_batch(self, metrics_data: List[dict], format_type: MetricFormat = MetricFormat.JSON,
						  tenant_id: str = None, source: str = "unknown") -> Dict[str, Any]:
		"""
		Ingest metrics batch with high throughput processing
		Returns ingestion results with success/failure counts
		"""
		start_time = time.time()
		tenant_id = tenant_id or "default"
		
		results = {
			'total': len(metrics_data),
			'successful': 0,
			'failed': 0,
			'errors': [],
			'batch_id': uuid7str(),
			'processing_time_ms': 0
		}
		
		try:
			# Rate limiting for batch
			async with self.rate_limiter:
				parsed_metrics = []
				parser = self.format_parsers.get(format_type)
				
				if not parser:
					results['failed'] = results['total']
					results['errors'].append(f"Unsupported format: {format_type}")
					return results
				
				# Parse all metrics
				for i, metric_data in enumerate(metrics_data):
					try:
						metric = await parser(metric_data, tenant_id, source)
						if metric:
							parsed_metrics.append(metric)
							results['successful'] += 1
						else:
							results['failed'] += 1
							results['errors'].append(f"Failed to parse metric at index {i}")
					except Exception as e:
						results['failed'] += 1
						results['errors'].append(f"Error at index {i}: {str(e)}")
				
				# Add to buffer in batch
				if parsed_metrics:
					await self.metrics_buffer.add_batch(parsed_metrics)
					
					# Create batch for processing
					batch = MetricsBatch(
						batch_id=results['batch_id'],
						metrics=parsed_metrics,
						format_type=format_type,
						created_at=datetime.utcnow(),
						tenant_id=tenant_id,
						source=source
					)
					
					# Queue batch for processing
					await self._queue_batch_for_processing(batch)
				
				# Update statistics
				for _ in range(results['successful']):
					await self._update_ingestion_stats(format_type, tenant_id, start_time)
				
				results['processing_time_ms'] = (time.time() - start_time) * 1000
				
		except Exception as e:
			results['failed'] = results['total']
			results['successful'] = 0
			results['errors'].append(f"Batch processing error: {str(e)}")
		
		return results
	
	async def stream_metrics(self, tenant_id: str = None, 
						    limit: int = 1000) -> AsyncGenerator[MonitoringMetric, None]:
		"""Stream recent metrics for real-time processing"""
		try:
			metrics = await self.metrics_buffer.get_recent_metrics(limit, tenant_id)
			for metric in metrics:
				yield metric
		except Exception as e:
			self._log_info(f"Error streaming metrics: {e}")
	
	async def get_ingestion_stats(self) -> dict:
		"""Get detailed ingestion statistics"""
		buffer_stats = self.metrics_buffer.get_stats()
		
		return {
			**self.ingestion_stats,
			'buffer_stats': buffer_stats,
			'processing_queues': {
				tenant: queue.qsize() for tenant, queue in self.processing_queues.items()
			},
			'running': self.running,
			'timestamp': datetime.utcnow().isoformat()
		}
	
	# Format parsers
	async def _parse_prometheus_format(self, data: dict, tenant_id: str, source: str) -> Optional[MonitoringMetric]:
		"""Parse Prometheus format metrics"""
		try:
			# Expected format: {"name": "metric_name", "value": 123.45, "labels": {"key": "value"}, "timestamp": 1642678800}
			return MonitoringMetric(
				tenant_id=tenant_id,
				name=data['name'],
				value=float(data['value']),
				labels=data.get('labels', {}),
				source=source,
				metric_type=MetricType.GAUGE,
				timestamp=datetime.fromtimestamp(data.get('timestamp', time.time()))
			)
		except (KeyError, ValueError, TypeError) as e:
			self._log_info(f"Error parsing Prometheus format: {e}")
			return None
	
	async def _parse_opentelemetry_format(self, data: dict, tenant_id: str, source: str) -> Optional[MonitoringMetric]:
		"""Parse OpenTelemetry format metrics"""
		try:
			# OpenTelemetry metric format
			return MonitoringMetric(
				tenant_id=tenant_id,
				name=data['name'],
				value=float(data['value']),
				labels=data.get('attributes', {}),
				source=source,
				metric_type=MetricType(data.get('type', 'gauge')),
				unit=data.get('unit'),
				timestamp=datetime.fromtimestamp(data.get('timestamp', time.time()) / 1000)
			)
		except (KeyError, ValueError, TypeError) as e:
			self._log_info(f"Error parsing OpenTelemetry format: {e}")
			return None
	
	async def _parse_statsd_format(self, data: dict, tenant_id: str, source: str) -> Optional[MonitoringMetric]:
		"""Parse StatsD format metrics"""
		try:
			# StatsD format: {"metric": "name", "value": 123, "type": "c|g|h|ms"}
			type_mapping = {
				'c': MetricType.COUNTER,
				'g': MetricType.GAUGE,
				'h': MetricType.HISTOGRAM,
				'ms': MetricType.HISTOGRAM
			}
			
			return MonitoringMetric(
				tenant_id=tenant_id,
				name=data['metric'],
				value=float(data['value']),
				labels=data.get('tags', {}),
				source=source,
				metric_type=type_mapping.get(data.get('type', 'g'), MetricType.GAUGE)
			)
		except (KeyError, ValueError, TypeError) as e:
			self._log_info(f"Error parsing StatsD format: {e}")
			return None
	
	async def _parse_json_format(self, data: dict, tenant_id: str, source: str) -> Optional[MonitoringMetric]:
		"""Parse custom JSON format metrics"""
		try:
			return MonitoringMetric(
				tenant_id=tenant_id,
				source=source,
				**{k: v for k, v in data.items() if k != 'tenant_id'}
			)
		except Exception as e:
			self._log_info(f"Error parsing JSON format: {e}")
			return None
	
	async def _parse_custom_format(self, data: dict, tenant_id: str, source: str) -> Optional[MonitoringMetric]:
		"""Parse custom format metrics - extensible for future formats"""
		try:
			# Custom parsing logic can be implemented here
			return MonitoringMetric(
				tenant_id=tenant_id,
				name=str(data.get('name', 'unknown')),
				value=float(data.get('value', 0)),
				labels=data.get('labels', {}),
				source=source
			)
		except Exception as e:
			self._log_info(f"Error parsing custom format: {e}")
			return None
	
	async def _queue_batch_for_processing(self, batch: MetricsBatch) -> None:
		"""Queue batch for background processing"""
		try:
			# For now, we'll process synchronously
			# In production, this would queue to a proper message queue
			await self._process_metrics_batch(batch)
		except Exception as e:
			self._log_info(f"Error queueing batch for processing: {e}")
	
	async def _process_metrics_batch(self, batch: MetricsBatch) -> None:
		"""Process metrics batch - placeholder for actual storage/analytics"""
		try:
			# This is where we would:
			# 1. Store metrics in time-series database
			# 2. Update analytics aggregations  
			# 3. Check alert rules
			# 4. Update performance statistics
			
			processing_time = time.time()
			
			# Simulate processing
			await asyncio.sleep(0.001)  # Minimal processing delay
			
			processing_duration = (time.time() - processing_time) * 1000
			self._log_info(f"Processed batch {batch.batch_id} with {batch.size()} metrics in {processing_duration:.2f}ms")
			
		except Exception as e:
			self._log_info(f"Error processing batch {batch.batch_id}: {e}")
	
	async def _batch_processor_loop(self, processor_name: str) -> None:
		"""Background batch processor loop"""
		self._log_info(f"Started batch processor: {processor_name}")
		
		try:
			while self.running:
				# Process queued metrics from all tenants
				for tenant_id, queue in self.processing_queues.items():
					if queue.empty():
						continue
					
					# Collect metrics for batching
					batch_metrics = []
					batch_start = time.time()
					
					# Collect up to batch_size metrics or timeout
					while (len(batch_metrics) < self.batch_size and 
						   (time.time() - batch_start) < self.batch_timeout_seconds):
						try:
							metric = await asyncio.wait_for(queue.get(), timeout=0.1)
							batch_metrics.append(metric)
						except asyncio.TimeoutError:
							break
					
					# Process batch if we have metrics
					if batch_metrics:
						batch = MetricsBatch(
							batch_id=uuid7str(),
							metrics=batch_metrics,
							format_type=MetricFormat.JSON,
							created_at=datetime.utcnow(),
							tenant_id=tenant_id,
							source="batch_processor"
						)
						await self._process_metrics_batch(batch)
				
				# Brief sleep to prevent busy waiting
				await asyncio.sleep(0.01)
				
		except asyncio.CancelledError:
			self._log_info(f"Batch processor {processor_name} cancelled")
		except Exception as e:
			self._log_info(f"Error in batch processor {processor_name}: {e}")
	
	async def _performance_monitor_loop(self) -> None:
		"""Monitor and update performance statistics"""
		try:
			while self.running:
				await asyncio.sleep(60)  # Update every minute
				
				# Reset rate calculations
				now = datetime.utcnow()
				time_delta = (now - self.ingestion_stats['last_reset']).total_seconds()
				
				if time_delta > 0:
					# Calculate ingestion rate
					current_total = self.ingestion_stats['total_ingested']
					self.ingestion_stats['ingestion_rate'] = current_total / time_delta
					
					# Reset counters for next interval
					self.ingestion_stats['last_reset'] = now
					
					self._log_info(f"Performance: {self.ingestion_stats['ingestion_rate']:.2f} metrics/sec")
				
		except asyncio.CancelledError:
			pass
		except Exception as e:
			self._log_info(f"Error in performance monitor: {e}")
	
	async def _update_ingestion_stats(self, format_type: MetricFormat, tenant_id: str, start_time: float) -> None:
		"""Update ingestion performance statistics"""
		latency_ms = (time.time() - start_time) * 1000
		
		# Update counters
		self.ingestion_stats['total_ingested'] += 1
		self.ingestion_stats['format_distribution'][format_type.value] += 1
		self.ingestion_stats['tenant_distribution'][tenant_id] += 1
		
		# Update rolling average latency
		current_avg = self.ingestion_stats['avg_latency_ms']
		self.ingestion_stats['avg_latency_ms'] = (current_avg * 0.9) + (latency_ms * 0.1)


# Factory function
def create_metrics_engine(config: dict = None) -> MetricsCollectionEngine:
	"""Create and configure metrics collection engine"""
	return MetricsCollectionEngine(config)