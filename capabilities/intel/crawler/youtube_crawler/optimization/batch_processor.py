"""
Batch Processor Module
======================

Efficient batch processing for YouTube API requests with intelligent batching,
load balancing, and parallel processing capabilities.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStrategy(Enum):
    """Batch processing strategies."""
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


class BatchPriority(Enum):
    """Batch priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 50
    min_batch_size: int = 1
    batch_timeout: float = 5.0
    max_concurrent_batches: int = 5
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    enable_priority: bool = True
    auto_flush_interval: float = 10.0
    max_queue_size: int = 1000
    retry_failed_items: bool = True
    max_item_retries: int = 3


@dataclass
class BatchItem(Generic[T]):
    """Item to be processed in a batch."""
    data: T
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0
    item_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.created_at < other.created_at  # FIFO for same priority


@dataclass
class BatchResult(Generic[T, R]):
    """Result of batch processing."""
    success: bool
    processed_items: List[T]
    results: List[R]
    failed_items: List[T]
    errors: List[str]
    batch_size: int
    processing_time: float
    start_time: float
    end_time: float
    batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor(Generic[T, R]):
    """
    Advanced batch processor with multiple strategies and optimizations.
    """
    
    def __init__(self, config: BatchConfig, processor_func: Optional[Callable] = None):
        self.config = config
        self.processor_func = processor_func
        self.queue: List[BatchItem[T]] = []
        self.processing_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        
        # State management
        self.running = False
        self.background_task = None
        self.last_flush = time.time()
        self.batch_counter = 0
        
        # Statistics
        self.stats = {
            'total_items_processed': 0,
            'total_batches_processed': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_processing_time': 0.0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'items_by_priority': defaultdict(int),
            'retry_distribution': defaultdict(int),
            'queue_size_history': deque(maxlen=100),
            'throughput_history': deque(maxlen=100)
        }
        
        # Adaptive parameters
        self.adaptive_batch_size = config.max_batch_size
        self.recent_performance = deque(maxlen=10)
        self.lock = asyncio.Lock()
    
    async def start(self):
        """Start the batch processor."""
        if self.running:
            return
        
        self.running = True
        self.background_task = asyncio.create_task(self._background_processor())
        logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processor and flush remaining items."""
        if not self.running:
            return
        
        self.running = False
        
        # Process remaining items
        await self.flush()
        
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Batch processor stopped")
    
    async def add_item(self, item: T, priority: BatchPriority = BatchPriority.NORMAL,
                      item_id: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """Add an item to the processing queue."""
        if not self.running:
            await self.start()
        
        batch_item = BatchItem(
            data=item,
            priority=priority,
            item_id=item_id,
            metadata=metadata or {}
        )
        
        async with self.lock:
            if len(self.queue) >= self.config.max_queue_size:
                logger.warning("Queue is full, dropping item")
                return False
            
            # Insert in priority order
            if self.config.enable_priority:
                heapq.heappush(self.queue, batch_item)
            else:
                self.queue.append(batch_item)
            
            self.stats['items_by_priority'][priority.name] += 1
            self.stats['queue_size_history'].append(len(self.queue))
        
        # Check if we should flush immediately
        await self._check_flush_conditions()
        
        return True
    
    async def add_items(self, items: List[T], priority: BatchPriority = BatchPriority.NORMAL) -> int:
        """Add multiple items to the processing queue."""
        added_count = 0
        
        for item in items:
            if await self.add_item(item, priority):
                added_count += 1
            else:
                break
        
        return added_count
    
    async def flush(self) -> List[BatchResult[T, R]]:
        """Flush all pending items and return results."""
        if not self.queue:
            return []
        
        results = []
        
        async with self.lock:
            while self.queue:
                batch_items = self._create_batch()
                if batch_items:
                    result = await self._process_batch(batch_items)
                    results.append(result)
        
        self.last_flush = time.time()
        return results
    
    async def process_immediately(self, items: List[T]) -> BatchResult[T, R]:
        """Process items immediately without queuing."""
        batch_items = [BatchItem(data=item) for item in items]
        return await self._process_batch(batch_items)
    
    async def _background_processor(self):
        """Background task to process batches continuously."""
        while self.running:
            try:
                # Check if we should flush based on time
                current_time = time.time()
                time_since_flush = current_time - self.last_flush
                
                if time_since_flush >= self.config.auto_flush_interval:
                    await self.flush()
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_flush_conditions(self):
        """Check if flush conditions are met."""
        should_flush = False
        
        async with self.lock:
            queue_size = len(self.queue)
            
            if self.config.strategy == BatchStrategy.SIZE_BASED:
                should_flush = queue_size >= self.config.max_batch_size
            
            elif self.config.strategy == BatchStrategy.TIME_BASED:
                time_since_flush = time.time() - self.last_flush
                should_flush = time_since_flush >= self.config.batch_timeout
            
            elif self.config.strategy == BatchStrategy.ADAPTIVE:
                should_flush = queue_size >= self.adaptive_batch_size
            
            elif self.config.strategy == BatchStrategy.PRIORITY_BASED:
                # Check for urgent items or size threshold
                urgent_items = sum(1 for item in self.queue 
                                 if item.priority == BatchPriority.URGENT)
                should_flush = (urgent_items > 0 or 
                              queue_size >= self.config.max_batch_size)
        
        if should_flush:
            asyncio.create_task(self._flush_single_batch())
    
    async def _flush_single_batch(self):
        """Flush a single batch."""
        async with self.semaphore:
            async with self.lock:
                batch_items = self._create_batch()
            
            if batch_items:
                result = await self._process_batch(batch_items)
                await self._update_adaptive_parameters(result)
    
    def _create_batch(self) -> List[BatchItem[T]]:
        """Create a batch from the queue."""
        if not self.queue:
            return []
        
        batch_size = self._determine_batch_size()
        batch_items = []
        
        # Extract items from queue
        for _ in range(min(batch_size, len(self.queue))):
            if self.config.enable_priority:
                item = heapq.heappop(self.queue)
            else:
                item = self.queue.pop(0)
            batch_items.append(item)
        
        return batch_items
    
    def _determine_batch_size(self) -> int:
        """Determine optimal batch size based on strategy."""
        queue_size = len(self.queue)
        
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            return min(self.adaptive_batch_size, queue_size)
        elif self.config.strategy == BatchStrategy.PRIORITY_BASED:
            # Process urgent items in smaller batches
            urgent_count = sum(1 for item in self.queue 
                             if item.priority == BatchPriority.URGENT)
            if urgent_count > 0:
                return min(urgent_count, self.config.max_batch_size // 2)
        
        return min(self.config.max_batch_size, queue_size)
    
    async def _process_batch(self, batch_items: List[BatchItem[T]]) -> BatchResult[T, R]:
        """Process a batch of items."""
        batch_id = f"batch_{self.batch_counter}"
        self.batch_counter += 1
        
        start_time = time.time()
        processed_items = []
        results = []
        failed_items = []
        errors = []
        
        try:
            # Extract data from batch items
            items_data = [item.data for item in batch_items]
            
            # Process the batch
            if self.processor_func:
                if asyncio.iscoroutinefunction(self.processor_func):
                    batch_results = await self.processor_func(items_data)
                else:
                    batch_results = self.processor_func(items_data)
                
                # Handle results
                if isinstance(batch_results, list):
                    results = batch_results
                    processed_items = items_data
                else:
                    results = [batch_results]
                    processed_items = items_data
            else:
                # Default processing (just return the items)
                results = items_data
                processed_items = items_data
            
            success = True
            
        except Exception as e:
            logger.error(f"Batch processing failed for {batch_id}: {e}")
            success = False
            failed_items = [item.data for item in batch_items]
            errors = [str(e)]
            
            # Handle retries for failed items
            if self.config.retry_failed_items:
                await self._handle_failed_items(batch_items, str(e))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update statistics
        await self._update_stats(batch_items, success, processing_time)
        
        return BatchResult(
            success=success,
            processed_items=processed_items,
            results=results,
            failed_items=failed_items,
            errors=errors,
            batch_size=len(batch_items),
            processing_time=processing_time,
            start_time=start_time,
            end_time=end_time,
            batch_id=batch_id
        )
    
    async def _handle_failed_items(self, batch_items: List[BatchItem[T]], error: str):
        """Handle failed items with retry logic."""
        retry_items = []
        
        for item in batch_items:
            item.retry_count += 1
            self.stats['retry_distribution'][item.retry_count] += 1
            
            if item.retry_count <= self.config.max_item_retries:
                # Add back to queue with lower priority
                item.priority = BatchPriority.LOW
                retry_items.append(item)
            else:
                logger.warning(f"Item {item.item_id} exceeded max retries, dropping")
        
        if retry_items:
            async with self.lock:
                for item in retry_items:
                    if self.config.enable_priority:
                        heapq.heappush(self.queue, item)
                    else:
                        self.queue.append(item)
    
    async def _update_stats(self, batch_items: List[BatchItem[T]], success: bool, processing_time: float):
        """Update processing statistics."""
        batch_size = len(batch_items)
        
        self.stats['total_items_processed'] += batch_size
        self.stats['total_batches_processed'] += 1
        self.stats['total_processing_time'] += processing_time
        
        if success:
            self.stats['successful_batches'] += 1
        else:
            self.stats['failed_batches'] += 1
        
        # Update averages
        total_batches = self.stats['total_batches_processed']
        self.stats['avg_batch_size'] = self.stats['total_items_processed'] / total_batches
        self.stats['avg_processing_time'] = self.stats['total_processing_time'] / total_batches
        
        # Update throughput
        throughput = batch_size / processing_time if processing_time > 0 else 0
        self.stats['throughput_history'].append(throughput)
    
    async def _update_adaptive_parameters(self, result: BatchResult[T, R]):
        """Update adaptive parameters based on batch performance."""
        if self.config.strategy != BatchStrategy.ADAPTIVE:
            return
        
        # Calculate performance score
        throughput = result.batch_size / result.processing_time if result.processing_time > 0 else 0
        performance_score = throughput if result.success else throughput * 0.1
        
        self.recent_performance.append(performance_score)
        
        if len(self.recent_performance) >= 5:
            # Adjust batch size based on recent performance
            recent_avg = sum(self.recent_performance) / len(self.recent_performance)
            
            if len(self.recent_performance) >= 10:
                older_avg = sum(list(self.recent_performance)[:5]) / 5
                
                if recent_avg > older_avg * 1.1:
                    # Performance improving, increase batch size
                    self.adaptive_batch_size = min(
                        self.config.max_batch_size,
                        int(self.adaptive_batch_size * 1.1)
                    )
                elif recent_avg < older_avg * 0.9:
                    # Performance degrading, decrease batch size
                    self.adaptive_batch_size = max(
                        self.config.min_batch_size,
                        int(self.adaptive_batch_size * 0.9)
                    )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive batch processor statistics."""
        current_queue_size = len(self.queue)
        
        # Calculate current throughput
        current_throughput = 0.0
        if self.stats['throughput_history']:
            current_throughput = sum(self.stats['throughput_history']) / len(self.stats['throughput_history'])
        
        return {
            'processor_state': {
                'running': self.running,
                'current_queue_size': current_queue_size,
                'adaptive_batch_size': self.adaptive_batch_size,
                'last_flush': time.time() - self.last_flush
            },
            'processing_stats': {
                'total_items_processed': self.stats['total_items_processed'],
                'total_batches_processed': self.stats['total_batches_processed'],
                'successful_batches': self.stats['successful_batches'],
                'failed_batches': self.stats['failed_batches'],
                'success_rate': (self.stats['successful_batches'] / max(1, self.stats['total_batches_processed'])),
                'avg_batch_size': self.stats['avg_batch_size'],
                'avg_processing_time': self.stats['avg_processing_time'],
                'current_throughput': current_throughput
            },
            'queue_stats': {
                'items_by_priority': dict(self.stats['items_by_priority']),
                'retry_distribution': dict(self.stats['retry_distribution']),
                'queue_size_trend': list(self.stats['queue_size_history'])[-10:],
                'throughput_trend': list(self.stats['throughput_history'])[-10:]
            },
            'config': {
                'max_batch_size': self.config.max_batch_size,
                'min_batch_size': self.config.min_batch_size,
                'batch_timeout': self.config.batch_timeout,
                'strategy': self.config.strategy.value,
                'max_concurrent_batches': self.config.max_concurrent_batches
            }
        }
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = {
            'total_items_processed': 0,
            'total_batches_processed': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_processing_time': 0.0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'items_by_priority': defaultdict(int),
            'retry_distribution': defaultdict(int),
            'queue_size_history': deque(maxlen=100),
            'throughput_history': deque(maxlen=100)
        }
        self.recent_performance.clear()
        self.adaptive_batch_size = self.config.max_batch_size
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Utility functions
def create_batch_config(preset: str = 'balanced') -> BatchConfig:
    """Create batch configuration with predefined presets."""
    presets = {
        'small_fast': BatchConfig(
            max_batch_size=10,
            batch_timeout=1.0,
            max_concurrent_batches=10,
            strategy=BatchStrategy.TIME_BASED
        ),
        'balanced': BatchConfig(
            max_batch_size=50,
            batch_timeout=5.0,
            max_concurrent_batches=5,
            strategy=BatchStrategy.ADAPTIVE
        ),
        'large_throughput': BatchConfig(
            max_batch_size=200,
            batch_timeout=10.0,
            max_concurrent_batches=3,
            strategy=BatchStrategy.SIZE_BASED
        ),
        'priority_aware': BatchConfig(
            max_batch_size=30,
            batch_timeout=3.0,
            max_concurrent_batches=8,
            strategy=BatchStrategy.PRIORITY_BASED,
            enable_priority=True
        )
    }
    
    return presets.get(preset, presets['balanced'])


async def process_items_in_batches(items: List[T], processor_func: Callable,
                                 config: Optional[BatchConfig] = None) -> List[BatchResult]:
    """Utility function to process items in batches."""
    config = config or BatchConfig()
    
    async with BatchProcessor(config, processor_func) as processor:
        await processor.add_items(items)
        return await processor.flush()


__all__ = [
    'BatchProcessor',
    'BatchConfig',
    'BatchResult',
    'BatchItem',
    'BatchStrategy',
    'BatchPriority',
    'create_batch_config',
    'process_items_in_batches'
]