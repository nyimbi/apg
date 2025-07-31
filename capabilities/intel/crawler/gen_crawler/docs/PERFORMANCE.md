# Gen Crawler Performance Guide

Comprehensive performance optimization guide for the gen_crawler package including benchmarking, tuning strategies, and scalability considerations.

## 🚀 Performance Overview

Gen Crawler is designed for high-performance web crawling with intelligent adaptation and resource optimization.

### Key Performance Features

- **Adaptive Strategy**: Automatic optimization based on site characteristics
- **Concurrent Processing**: Configurable parallelism with resource management
- **Intelligent Caching**: Multi-level caching for profiles and content
- **Memory Management**: Built-in memory limits and cleanup
- **Async Architecture**: Non-blocking I/O operations throughout

## 📊 Performance Benchmarks

### Standard Performance Targets

| Metric | Target | Typical Range |
|--------|--------|---------------|
| Pages per second | 2-5 | 1-10 |
| Memory usage per 1000 pages | < 100MB | 50-200MB |
| CPU usage (per core) | < 50% | 20-80% |
| Success rate | > 90% | 85-98% |
| Response time | < 30s | 5-60s |

### Benchmark Test Results

```python
# Example benchmark results
Site Type              Pages/sec  Memory(MB)  Success%  Avg Time(s)
-------------------------------------------------------------
News sites (adaptive)     3.2       85        94.2        2.1
Static sites (HTTP)        8.7       45        98.1        0.8
JS-heavy sites (Browser)   1.4      156        89.3        4.2
Mixed content (Adaptive)   4.1       92        91.7        2.8
```

## ⚙️ Configuration Tuning

### 1. Concurrency Settings

**Basic Tuning**:
```python
# Conservative (safe for most sites)
config.settings.performance.max_concurrent = 3
config.settings.performance.crawl_delay = 2.0

# Moderate (balanced performance)
config.settings.performance.max_concurrent = 5
config.settings.performance.crawl_delay = 1.5

# Aggressive (maximum performance)
config.settings.performance.max_concurrent = 10
config.settings.performance.crawl_delay = 1.0
```

**Site-Specific Tuning**:
```python
# News sites (respectful crawling)
news_config = {
    'max_concurrent': 3,
    'crawl_delay': 3.0,
    'request_timeout': 30
}

# Static sites (faster crawling)
static_config = {
    'max_concurrent': 8,
    'crawl_delay': 1.0,
    'request_timeout': 15
}

# E-commerce sites (careful crawling)
ecommerce_config = {
    'max_concurrent': 2,
    'crawl_delay': 5.0,
    'request_timeout': 45
}
```

### 2. Memory Optimization

**Memory-Efficient Settings**:
```python
config.settings.performance.memory_limit_mb = 512
config.settings.performance.max_pages_per_site = 100
config.settings.save_raw_html = False  # Saves memory
config.settings.compression_enabled = True  # Compress stored data
```

**Memory Monitoring**:
```python
import psutil
import os

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

# Example memory tracking
initial_memory = monitor_memory_usage()
# ... crawling operations ...
final_memory = monitor_memory_usage()
print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
```

### 3. Content Processing Optimization

**Fast Content Processing**:
```python
# Use faster extraction methods
config.settings.extraction_method = 'beautifulsoup'  # Faster than trafilatura

# Disable expensive analysis
config.settings.enable_image_extraction = False
config.settings.enable_link_analysis = False

# Set content limits
config.settings.content_filters.max_content_length = 100000  # 100KB limit
```

**Quality vs Speed Trade-offs**:
```python
# High quality (slower)
quality_config = {
    'extraction_method': 'trafilatura',
    'enable_content_analysis': True,
    'enable_image_extraction': True,
    'min_content_length': 200
}

# High speed (lower quality)
speed_config = {
    'extraction_method': 'beautifulsoup',
    'enable_content_analysis': False,
    'enable_image_extraction': False,
    'min_content_length': 50
}
```

## 🧠 Adaptive Performance Optimization

### 1. Automatic Strategy Selection

The adaptive crawler automatically optimizes performance based on site characteristics:

```python
class AdaptiveOptimizer:
    def optimize_for_site(self, site_profile: SiteProfile) -> Dict[str, Any]:
        """
        Automatic optimization based on site profile:
        
        Fast sites (< 2s avg):
        - Increase concurrency to 8
        - Reduce delay to 1.0s
        - Use HTTP-only strategy
        
        Slow sites (> 5s avg):
        - Reduce concurrency to 2
        - Increase delay to 3.0s
        - Use browser strategy if needed
        
        Error-prone sites (< 80% success):
        - Reduce concurrency to 1
        - Increase delay to 5.0s
        - Add extra retries
        """
```

### 2. Performance-Based Strategy Switching

```python
# Strategy performance thresholds
STRATEGY_THRESHOLDS = {
    'http_only': {
        'min_success_rate': 85.0,
        'max_avg_time': 3.0,
        'preferred_concurrency': 8
    },
    'browser_only': {
        'min_success_rate': 70.0,
        'max_avg_time': 8.0, 
        'preferred_concurrency': 2
    },
    'adaptive': {
        'min_success_rate': 80.0,
        'max_avg_time': 5.0,
        'preferred_concurrency': 5
    }
}
```

### 3. Real-time Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'pages_per_second': 0.0,
            'memory_usage_mb': 0.0,
            'success_rate': 0.0,
            'avg_response_time': 0.0
        }
    
    def update_metrics(self, crawl_result: GenSiteResult):
        """Update performance metrics in real-time."""
        
    def should_adjust_concurrency(self) -> bool:
        """Determine if concurrency should be adjusted."""
        
    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get performance-optimized settings."""
```

## 🏎️ High-Performance Scenarios

### 1. Bulk Site Crawling

**Configuration for crawling 100+ sites**:
```python
bulk_config = create_gen_config()
bulk_config.settings.performance.max_pages_per_site = 50  # Limit per site
bulk_config.settings.performance.max_concurrent = 3       # Conservative
bulk_config.settings.performance.crawl_delay = 2.0       # Respectful
bulk_config.settings.performance.request_timeout = 20    # Faster timeout

# Use batch processing
async def crawl_sites_in_batches(sites: List[str], batch_size: int = 10):
    results = []
    for i in range(0, len(sites), batch_size):
        batch = sites[i:i + batch_size]
        batch_results = await crawl_multiple_sites(batch, bulk_config)
        results.extend(batch_results)
        
        # Cleanup between batches
        await cleanup_resources()
        await asyncio.sleep(5)  # Pause between batches
    
    return results
```

### 2. Real-time Monitoring Setup

**Configuration for continuous monitoring**:
```python
monitoring_config = create_gen_config()
monitoring_config.settings.performance.max_pages_per_site = 20
monitoring_config.settings.performance.crawl_delay = 10.0  # Very respectful
monitoring_config.settings.adaptive.enable_adaptive_crawling = True

# Continuous monitoring with backoff
async def continuous_monitoring(sites: List[str], interval: int = 300):
    while True:
        try:
            for site in sites:
                result = await crawl_site(site, monitoring_config)
                await process_monitoring_result(result)
                
            await asyncio.sleep(interval)
            
        except Exception as e:
            # Exponential backoff on errors
            interval = min(interval * 2, 3600)  # Max 1 hour
            await asyncio.sleep(interval)
```

### 3. Large-Scale Data Processing

**Configuration for processing thousands of pages**:
```python
large_scale_config = {
    'performance': {
        'max_pages_per_site': 1000,
        'max_concurrent': 5,
        'memory_limit_mb': 2048,
        'enable_disk_cache': True
    },
    'content_filters': {
        'enable_content_deduplication': True,
        'max_content_length': 500000  # 500KB limit
    },
    'database': {
        'enable_database': True,
        'batch_size': 100,  # Batch database inserts
        'connection_pool_size': 20
    }
}
```

## 📈 Performance Optimization Strategies

### 1. Content Processing Optimization

**Extraction Method Selection**:
```python
# Performance ranking (fastest to slowest)
EXTRACTION_METHODS = [
    'beautifulsoup',    # Fastest, good quality
    'readability',      # Fast, clean content
    'newspaper',        # Medium, article-focused
    'trafilatura',      # Slower, highest quality
]

def select_optimal_extractor(site_profile: SiteProfile) -> str:
    """Select extractor based on site characteristics."""
    if site_profile.requires_javascript:
        return 'newspaper'  # Better for complex sites
    elif site_profile.success_rate > 95:
        return 'beautifulsoup'  # Fast for reliable sites
    else:
        return 'trafilatura'  # Quality for difficult sites
```

**Content Filtering Optimization**:
```python
def optimize_content_filters(site_type: str) -> Dict[str, Any]:
    """Optimize filters based on site type."""
    
    if site_type == 'news':
        return {
            'include_patterns': ['article', 'story', 'news'],
            'exclude_patterns': ['comment', 'advertisement'],
            'min_content_length': 300
        }
    elif site_type == 'blog':
        return {
            'include_patterns': ['post', 'blog', 'article'],
            'exclude_patterns': ['sidebar', 'widget'],
            'min_content_length': 200
        }
    elif site_type == 'corporate':
        return {
            'include_patterns': ['about', 'service', 'product'],
            'exclude_patterns': ['legal', 'privacy'],
            'min_content_length': 100
        }
```

### 2. Network Optimization

**Connection Management**:
```python
# Optimize for different network conditions
NETWORK_CONFIGS = {
    'fast': {
        'max_concurrent': 10,
        'request_timeout': 10,
        'max_retries': 2,
        'crawl_delay': 0.5
    },
    'medium': {
        'max_concurrent': 5,
        'request_timeout': 20,
        'max_retries': 3,
        'crawl_delay': 1.5
    },
    'slow': {
        'max_concurrent': 2,
        'request_timeout': 45,
        'max_retries': 5,
        'crawl_delay': 3.0
    }
}
```

**Error Handling Optimization**:
```python
class OptimizedErrorHandler:
    def __init__(self):
        self.error_patterns = {
            'rate_limited': ['429', 'rate limit', 'too many requests'],
            'blocked': ['403', 'forbidden', 'access denied'],
            'timeout': ['timeout', 'connection reset'],
            'not_found': ['404', 'not found']
        }
    
    def get_retry_strategy(self, error_type: str) -> Dict[str, Any]:
        """Get optimized retry strategy based on error type."""
        if error_type == 'rate_limited':
            return {'delay': 60, 'backoff': 2.0, 'max_retries': 3}
        elif error_type == 'timeout':
            return {'delay': 5, 'backoff': 1.5, 'max_retries': 2}
        else:
            return {'delay': 10, 'backoff': 1.0, 'max_retries': 1}
```

### 3. Resource Management

**Memory Management**:
```python
class MemoryManager:
    def __init__(self, limit_mb: int = 1024):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.current_usage = 0
    
    def check_memory_usage(self):
        """Check if memory usage is within limits."""
        process = psutil.Process()
        self.current_usage = process.memory_info().rss
        return self.current_usage < self.limit_bytes
    
    def cleanup_if_needed(self):
        """Cleanup memory if approaching limits."""
        if not self.check_memory_usage():
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear caches
            self.clear_caches()
```

**CPU Optimization**:
```python
def optimize_cpu_usage(cpu_count: int) -> Dict[str, Any]:
    """Optimize settings based on available CPU cores."""
    
    # Conservative: 1 thread per 2 cores
    conservative_concurrency = max(1, cpu_count // 2)
    
    # Aggressive: 1.5 threads per core
    aggressive_concurrency = int(cpu_count * 1.5)
    
    return {
        'conservative': {
            'max_concurrent': conservative_concurrency,
            'crawl_delay': 2.0
        },
        'aggressive': {
            'max_concurrent': aggressive_concurrency,
            'crawl_delay': 1.0
        }
    }
```

## 🔍 Performance Monitoring & Profiling

### 1. Built-in Performance Metrics

```python
class PerformanceProfiler:
    def __init__(self):
        self.metrics = {
            'crawl_start_time': None,
            'pages_processed': 0,
            'total_content_size': 0,
            'extraction_times': [],
            'network_times': [],
            'processing_times': []
        }
    
    def record_page_processing(self, page_result: GenCrawlResult):
        """Record performance metrics for a page."""
        self.metrics['pages_processed'] += 1
        self.metrics['total_content_size'] += len(page_result.content)
        self.metrics['extraction_times'].append(page_result.crawl_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'pages_per_second': self._calculate_pages_per_second(),
            'avg_page_size': self._calculate_avg_page_size(),
            'avg_extraction_time': self._calculate_avg_extraction_time(),
            'memory_efficiency': self._calculate_memory_efficiency()
        }
```

### 2. External Profiling Tools

**CPU Profiling**:
```python
# Using cProfile for CPU profiling
import cProfile
import pstats

def profile_crawl_performance():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run crawling operation
    result = await crawl_site("https://example.com")
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

**Memory Profiling**:
```python
# Using memory_profiler
from memory_profiler import profile

@profile
async def memory_profiled_crawl():
    crawler = GenCrawler(config)
    await crawler.initialize()
    result = await crawler.crawl_site("https://example.com")
    await crawler.cleanup()
    return result
```

### 3. Real-time Performance Dashboard

```python
class PerformanceDashboard:
    def __init__(self):
        self.real_time_metrics = {
            'current_pages_per_second': 0.0,
            'current_memory_usage': 0.0,
            'current_cpu_usage': 0.0,
            'success_rate_last_100': 0.0,
            'avg_response_time_last_100': 0.0
        }
    
    def update_real_time_metrics(self):
        """Update metrics in real-time."""
        
    def should_scale_up(self) -> bool:
        """Determine if scaling up would improve performance."""
        
    def should_scale_down(self) -> bool:
        """Determine if scaling down would improve stability."""
```

## ⚡ Performance Best Practices

### 1. Configuration Best Practices

```python
# DO: Use site-specific configurations
news_config = create_news_optimized_config()
static_config = create_static_optimized_config()

# DON'T: Use one-size-fits-all configuration
# bad_config = create_generic_config()

# DO: Enable adaptive crawling for unknown sites
config.settings.adaptive.enable_adaptive_crawling = True

# DO: Set reasonable resource limits
config.settings.performance.memory_limit_mb = 1024
config.settings.performance.max_pages_per_site = 500

# DON'T: Set unlimited resources
# config.settings.performance.max_pages_per_site = float('inf')
```

### 2. Implementation Best Practices

```python
# DO: Use async/await consistently
async def process_sites(sites: List[str]):
    tasks = [crawl_site(site) for site in sites]
    results = await asyncio.gather(*tasks, return_exceptions=True)

# DON'T: Mix sync and async code
# def bad_process_sites(sites):
#     return [sync_crawl_site(site) for site in sites]

# DO: Implement proper resource cleanup
async def safe_crawl_with_cleanup():
    crawler = None
    try:
        crawler = GenCrawler(config)
        await crawler.initialize()
        return await crawler.crawl_site(url)
    finally:
        if crawler:
            await crawler.cleanup()

# DO: Use batching for large datasets
async def process_large_dataset(urls: List[str], batch_size: int = 10):
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        await process_batch(batch)
        await asyncio.sleep(1)  # Brief pause between batches
```

### 3. Monitoring Best Practices

```python
# DO: Monitor key performance indicators
def monitor_performance(crawler: GenCrawler):
    stats = crawler.get_statistics()
    
    # Alert on poor performance
    if stats['success_rate'] < 80:
        logger.warning(f"Low success rate: {stats['success_rate']}%")
    
    if stats['pages_per_second'] < 1:
        logger.warning(f"Slow crawling: {stats['pages_per_second']} pages/sec")

# DO: Implement performance-based auto-scaling
async def auto_scale_based_on_performance(crawler: GenCrawler):
    metrics = crawler.get_performance_metrics()
    
    if metrics['cpu_usage'] < 30 and metrics['memory_usage'] < 50:
        # Scale up
        crawler.config['max_concurrent'] *= 1.5
    elif metrics['cpu_usage'] > 80 or metrics['memory_usage'] > 90:
        # Scale down
        crawler.config['max_concurrent'] *= 0.75
```

This performance guide provides comprehensive strategies for optimizing gen_crawler performance across different scenarios and use cases, ensuring efficient and scalable web crawling operations.