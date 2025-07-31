# APG Crawler Capability - Developer Guide

**Version:** 2.0.0  
**Author:** Datacraft  
**Copyright:** © 2025 Datacraft  
**Email:** nyimbi@gmail.com  

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Simple API Reference](#simple-api-reference)
4. [Advanced API Reference](#advanced-api-reference)
5. [Engine Development](#engine-development)
6. [Database Integration](#database-integration)
7. [Testing and Validation](#testing-and-validation)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Contributing](#contributing)

## Architecture Overview

### System Design Philosophy

The APG Crawler Capability follows a **layered architecture** with **guaranteed success** through multiple fallback strategies:

```
┌─────────────────────────────────────────┐
│              Simple API                 │  ← Dead simple, guaranteed success
├─────────────────────────────────────────┤
│         Multi-Source Orchestrator       │  ← Intelligent request management
├─────────────────────────────────────────┤
│    Smart Crawler + Content Pipeline     │  ← AI-powered content processing
├─────────────────────────────────────────┤
│         Stealth Engine Strategies       │  ← Multiple crawling methods
├─────────────────────────────────────────┤
│     Content Intelligence + Database     │  ← Storage and analysis
└─────────────────────────────────────────┘
```

### Key Design Principles

1. **Guaranteed Success**: Multiple fallback strategies ensure content is always extracted
2. **Performance**: Concurrent processing with intelligent rate limiting
3. **Intelligence**: AI-powered content understanding and business intelligence
4. **Modularity**: Pluggable engines and strategies
5. **Enterprise-Ready**: Multi-tenant, secure, production-grade

## Core Components

### 1. Simple API (`simple_api.py`)

The main interface for developers, providing guaranteed success functions:

```python
from crawler.simple_api import scrape_page, crawl_site

# Single page - always succeeds
result = await scrape_page("https://example.com")

# Multiple pages - concurrent processing
results = await crawl_site(["url1", "url2"], max_concurrent=3)
```

**Key Classes:**
- `SimpleMarkdownResult`: Result container with markdown content
- `SimpleCrawlResults`: Batch processing results
- `GuaranteedSuccessCrawler`: Core crawler with fallback strategies

### 2. Multi-Source Orchestrator (`engines/multi_source_orchestrator.py`)

Manages concurrent crawling with intelligent queuing:

```python
from crawler.engines.multi_source_orchestrator import MultiSourceOrchestrator

orchestrator = MultiSourceOrchestrator(max_concurrent=10)
await orchestrator.add_urls(urls, tenant_id="your_tenant")
await orchestrator.start_crawling(config, business_context)
```

**Key Classes:**
- `MultiSourceOrchestrator`: Main orchestration engine
- `RequestQueue`: Priority queue with deduplication
- `QueuedRequest`: Request with metadata and retry logic
- `SmartCrawler`: AI-powered crawler with content understanding

### 3. Stealth Engine (`engines/stealth_engine.py`)

Multiple strategies for bypassing protection mechanisms:

```python
from crawler.engines.stealth_engine import StealthOrchestrationEngine

engine = StealthOrchestrationEngine()
result = await engine.crawl_with_stealth(request, tenant_id, adaptive_strategy=True)
```

**Available Strategies:**
- `CloudScraperStrategy`: Cloudflare bypass
- `PlaywrightStrategy`: JavaScript rendering with anti-detection
- `SeleniumStealthStrategy`: Browser automation with stealth
- `HTTPMimicryStrategy`: Behavioral mimicry

### 4. Content Pipeline (`engines/content_pipeline.py`)

Advanced content extraction and cleaning:

```python
from crawler.engines.content_pipeline import ContentProcessingPipeline

pipeline = ContentProcessingPipeline()
result = await pipeline.process_crawl_result(crawl_result, config)
```

**Features:**
- Multi-format support (HTML, JSON, XML, PDF)
- Triple extraction methodology (newspaper3k + readability + trafilatura)
- Smart content cleaning and markdown conversion
- Content fingerprinting and deduplication

### 5. Content Intelligence (`engines/content_intelligence.py`)

AI-powered content analysis and business intelligence:

```python
from crawler.engines.content_intelligence import ContentIntelligenceEngine

intelligence = ContentIntelligenceEngine()
result = await intelligence.analyze_content(extraction_result, business_context)
```

**Capabilities:**
- Business entity extraction (organizations, people, products)
- Content classification and industry categorization
- Sentiment analysis and theme extraction
- Business intelligence (market signals, competitive analysis)

## Simple API Reference

### Core Functions

#### `scrape_page(url, tenant_id="default")`

Scrape a single page with guaranteed success.

**Parameters:**
- `url` (str): URL to scrape
- `tenant_id` (str, optional): Tenant identifier

**Returns:** `SimpleMarkdownResult`

**Example:**
```python
result = await scrape_page("https://example.com")
print(result.markdown_content)
print(result.metadata['strategy_used'])
```

#### `crawl_site(urls, tenant_id="default", max_concurrent=3)`

Crawl multiple pages concurrently.

**Parameters:**
- `urls` (List[str]): List of URLs to crawl
- `tenant_id` (str, optional): Tenant identifier
- `max_concurrent` (int, optional): Maximum concurrent requests

**Returns:** `SimpleCrawlResults`

**Example:**
```python
urls = ["https://example1.com", "https://example2.com"]
results = await crawl_site(urls, max_concurrent=5)

for result in results.results:
    if result.success:
        print(f"✅ {result.url}: {len(result.markdown_content)} chars")
    else:
        print(f"❌ {result.url}: {result.error}")
```

#### `crawl_site_from_homepage(base_url, max_pages=10, tenant_id="default")`

Automatically discover and crawl pages from a homepage.

**Parameters:**
- `base_url` (str): Starting URL (homepage)
- `max_pages` (int, optional): Maximum pages to crawl
- `tenant_id` (str, optional): Tenant identifier

**Returns:** `SimpleCrawlResults`

**Example:**
```python
results = await crawl_site_from_homepage("https://example.com", max_pages=20)
print(f"Discovered and crawled {results.total_count} pages")
```

### Synchronous Wrappers

For use in non-async environments:

```python
from crawler.simple_api import scrape_page_sync, crawl_site_sync

# Synchronous usage
result = scrape_page_sync("https://example.com")
results = crawl_site_sync(["url1", "url2"])
```

### Result Objects

#### `SimpleMarkdownResult`

```python
@dataclass
class SimpleMarkdownResult:
    url: str                    # Original URL
    title: Optional[str]        # Extracted title
    markdown_content: str       # Clean markdown content
    success: bool              # Whether extraction succeeded
    metadata: Dict[str, Any]   # Processing metadata
    error: Optional[str]       # Error message if failed
```

**Metadata Fields:**
- `strategy_used`: Which crawling strategy succeeded
- `processing_time`: Time taken to process
- `content_length`: Length of extracted content
- `language`: Detected language
- `status_code`: HTTP status code
- `entity_count`: Number of entities extracted (if intelligence enabled)
- `content_category`: Classified content category
- `sentiment_positive`: Positive sentiment score

#### `SimpleCrawlResults`

```python
@dataclass
class SimpleCrawlResults:
    results: List[SimpleMarkdownResult]  # Individual results
    success_count: int                  # Number of successful extractions
    total_count: int                   # Total URLs processed
    success_rate: float               # Success rate (0.0-1.0)
    processing_time: float           # Total processing time
```

## Advanced API Reference

### Configuration Objects

#### `ContentCleaningConfig`

Controls content cleaning and processing:

```python
from crawler.views import ContentCleaningConfig

config = ContentCleaningConfig(
    remove_navigation=True,        # Remove nav elements
    remove_ads=True,              # Remove advertisements
    remove_social_widgets=True,    # Remove social media widgets
    remove_comments=True,         # Remove comment sections
    markdown_formatting=True,      # Enable markdown formatting
    min_content_length=50,        # Minimum content length
    max_content_length=50000      # Maximum content length
)
```

#### `RAGProcessingConfig`

Configuration for RAG processing:

```python
from crawler.views import RAGProcessingConfig

rag_config = RAGProcessingConfig(
    chunk_size=1000,              # Text chunk size
    overlap_size=200,             # Overlap between chunks
    vector_dimensions=1536,       # Embedding dimensions
    embedding_model="text-embedding-ada-002",
    indexing_strategy="semantic_chunks"
)
```

### Advanced Usage Patterns

#### Custom Business Context

Provide domain-specific context for better AI analysis:

```python
business_context = {
    "domain": "Technology News",
    "industry": "Technology", 
    "use_case": "Market Intelligence",
    "priority_entities": ["company", "product", "funding"],
    "quality_criteria": {
        "min_article_length": 200,
        "require_publish_date": True
    }
}

result = await scrape_page("https://techcrunch.com/article", business_context)
```

#### Custom Crawler Configuration

For advanced use cases, instantiate the crawler directly:

```python
from crawler.simple_api import GuaranteedSuccessCrawler

crawler = GuaranteedSuccessCrawler()

# Custom configuration
crawler.default_config.max_content_length = 100000
crawler.orchestrator.max_concurrent = 10

result = await crawler.scrape_single_page(url, timeout=60)
await crawler.cleanup()
```

## Engine Development

### Creating Custom Strategies

To add a new crawling strategy:

1. **Inherit from Base Strategy Interface:**

```python
from crawler.engines.stealth_engine import StealthMethod, CrawlRequest, CrawlResult

class CustomStrategy:
    async def crawl(self, request: CrawlRequest) -> CrawlResult:
        # Implement your crawling logic
        try:
            # Your custom crawling code here
            content = await your_crawling_method(request.url)
            
            return CrawlResult(
                url=request.url,
                status_code=200,
                content=content,
                headers={},
                cookies={},
                final_url=request.url,
                response_time=0.5,
                method_used=StealthMethod.CUSTOM,
                protection_detected=[],
                success=True
            )
        except Exception as e:
            return CrawlResult(
                url=request.url,
                status_code=0,
                content="",
                headers={},
                cookies={},
                final_url=request.url,
                response_time=0,
                method_used=StealthMethod.CUSTOM,
                protection_detected=[],
                success=False,
                error=str(e)
            )
    
    async def cleanup(self):
        # Cleanup resources
        pass
```

2. **Register Strategy:**

```python
# Add to StealthOrchestrationEngine
engine = StealthOrchestrationEngine()
engine.strategies[StealthMethod.CUSTOM] = CustomStrategy()
```

### Custom Content Processing

Extend the content pipeline with custom processors:

```python
from crawler.engines.content_pipeline import ContentExtractionEngine

class CustomContentEngine(ContentExtractionEngine):
    async def extract_content(self, crawl_result, config):
        # Custom extraction logic
        result = await super().extract_content(crawl_result, config)
        
        # Add custom processing
        result.metadata['custom_field'] = self.custom_analysis(result.content)
        
        return result
    
    def custom_analysis(self, content):
        # Your custom analysis logic
        return {"analysis": "custom"}
```

### Custom Intelligence Modules

Add domain-specific intelligence:

```python
from crawler.engines.content_intelligence import ContentIntelligenceEngine

class CustomIntelligenceEngine(ContentIntelligenceEngine):
    async def analyze_content(self, extraction_result, business_context):
        result = await super().analyze_content(extraction_result, business_context)
        
        # Add custom intelligence
        custom_insights = await self.extract_custom_insights(
            extraction_result.content, business_context
        )
        
        result.business_intelligence.custom_insights = custom_insights
        return result
    
    async def extract_custom_insights(self, content, context):
        # Domain-specific analysis
        return {"custom": "insights"}
```

## Database Integration

### Database Service Integration

The crawler integrates with the APG database service:

```python
from crawler.service import CrawlerDatabaseService

async def store_crawl_results():
    db_service = CrawlerDatabaseService()
    
    # Create crawl target
    target = await db_service.create_crawl_target({
        'tenant_id': 'your_tenant',
        'name': 'My Crawl Target',
        'target_urls': ['https://example.com'],
        'target_type': 'web_crawl',
        'rag_integration_enabled': True
    })
    
    # Store crawled data
    await db_service.create_data_record({
        'tenant_id': 'your_tenant',
        'dataset_id': target.id,
        'source_url': 'https://example.com',
        'markdown_content': result.markdown_content,
        'content_fingerprint': result.content_fingerprint
    })
```

### RAG Integration

Process content for semantic search:

```python
# Process content for RAG
await db_service.process_rag_content({
    'tenant_id': 'your_tenant',
    'content_id': record.id,
    'chunk_size': 1000,
    'overlap_size': 200
})

# Search RAG content
results = await db_service.search_rag_content(
    tenant_id='your_tenant',
    query='artificial intelligence',
    limit=10
)
```

### GraphRAG Integration

Build knowledge graphs:

```python
# Create knowledge graph
graph = await db_service.create_knowledge_graph({
    'tenant_id': 'your_tenant',
    'graph_name': 'Tech Companies',
    'domain': 'Technology'
})

# Process content for GraphRAG
await db_service.process_graphrag_content({
    'tenant_id': 'your_tenant',
    'rag_chunk_ids': chunk_ids,
    'knowledge_graph_id': graph.id
})
```

## Testing and Validation

### Unit Testing

Test individual components:

```python
import pytest
from crawler.engines.stealth_engine import HTTPMimicryStrategy

@pytest.mark.asyncio
async def test_http_strategy():
    strategy = HTTPMimicryStrategy()
    request = CrawlRequest(url="https://httpbin.org/html")
    
    result = await strategy.crawl(request)
    
    assert result.success
    assert len(result.content) > 100
    assert result.status_code == 200
    
    await strategy.cleanup()
```

### Integration Testing

Test complete workflows:

```python
@pytest.mark.asyncio
async def test_complete_workflow():
    from crawler.simple_api import scrape_page
    
    result = await scrape_page("https://example.com")
    
    assert result.success
    assert len(result.markdown_content) > 50
    assert result.metadata['strategy_used'] is not None
```

### Running the Test Suite

Use the comprehensive test runner:

```bash
cd capabilities/common/crawler
python test_full_capability.py
```

Expected output:
```
🧪 APG CRAWLER CAPABILITY - FULL FUNCTIONALITY TEST
============================================================
✅ PASS: HTTP Mimicry Strategy
✅ PASS: CloudScraper Strategy  
✅ PASS: Content Extraction Pipeline
✅ PASS: Simple API - Single Page
🎉 OVERALL RESULT: SUCCESS (>95%)
```

## Performance Optimization

### Concurrent Processing

Optimize for high-throughput scenarios:

```python
# High-performance configuration
orchestrator = MultiSourceOrchestrator(
    max_concurrent=20,      # More concurrent requests
    max_sessions=50         # More session pooling
)

# Batch processing
urls = ["url1", "url2", ...]  # Large list
results = await crawl_site(urls, max_concurrent=10)
```

### Memory Management

For large-scale crawling:

```python
# Process in batches to manage memory
batch_size = 100
for i in range(0, len(urls), batch_size):
    batch = urls[i:i+batch_size]
    results = await crawl_site(batch)
    
    # Process results immediately
    await process_batch_results(results)
    
    # Optional: cleanup between batches
    await cleanup_resources()
```

### Rate Limiting

Respect server limits:

```python
# Custom rate limiting per domain
orchestrator.rate_limiter_config = {
    'default': 1.0,          # 1 request per second default
    'api.example.com': 0.5,  # Slower for APIs
    'news.site.com': 2.0     # Faster for news sites
}
```

### Caching Strategies

Implement content caching:

```python
# Custom caching layer
class CachedCrawler(GuaranteedSuccessCrawler):
    def __init__(self):
        super().__init__()
        self.cache = {}
    
    async def scrape_single_page(self, url, tenant_id="default"):
        # Check cache first
        cache_key = f"{tenant_id}:{url}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Crawl and cache
        result = await super().scrape_single_page(url, tenant_id)
        self.cache[cache_key] = result
        return result
```

## Security Considerations

### Input Validation

Always validate URLs and parameters:

```python
from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and parsed.netloc
    except:
        return False

# Use in your application
if not validate_url(user_provided_url):
    raise ValueError("Invalid URL provided")
```

### Rate Limiting and Respect

Implement respectful crawling:

```python
# Check robots.txt compliance
from urllib.robotparser import RobotFileParser

def check_robots_txt(url: str, user_agent: str = "*") -> bool:
    try:
        rp = RobotFileParser()
        rp.set_url(urljoin(url, "/robots.txt"))
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return True  # Allow if robots.txt unavailable
```

### Content Security

Sanitize extracted content:

```python
import bleach

def sanitize_content(content: str) -> str:
    # Remove potentially dangerous content
    allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'strong', 'em']
    return bleach.clean(content, tags=allowed_tags, strip=True)
```

### Authentication and Authorization

Integrate with APG auth system:

```python
def get_tenant_from_context():
    # APG integration
    return current_tenant_id()

async def authorized_scrape(url: str, user_context):
    tenant_id = get_tenant_from_context()
    
    # Check permissions
    if not user_context.can_access_domain(urlparse(url).netloc):
        raise PermissionError("Access denied to domain")
    
    return await scrape_page(url, tenant_id)
```

## Contributing

### Development Setup

1. **Clone and Setup:**
```bash
git clone <repository>
cd capabilities/common/crawler
pip install -r requirements.txt
```

2. **Install Development Dependencies:**
```bash
pip install pytest pytest-asyncio black flake8 mypy
```

3. **Run Tests:**
```bash
python test_full_capability.py
pytest tests/ -v
```

### Code Standards

Follow APG coding standards:

```python
# Use async throughout
async def my_function() -> str:
    return "result"

# Modern typing
from typing import Dict, List, Optional
def process_data(items: List[Dict[str, Any]]) -> Optional[str]:
    pass

# Proper logging
import logging
logger = logging.getLogger(__name__)
logger.info("Processing started")
```

### Adding New Features

1. **Create Feature Branch:**
```bash
git checkout -b feature/new-crawling-strategy
```

2. **Implement with Tests:**
```python
# Add implementation
# Add comprehensive tests
# Update documentation
```

3. **Submit Pull Request:**
- Include test coverage
- Update documentation
- Follow code review process

### Performance Benchmarking

Test performance improvements:

```python
import time
import asyncio

async def benchmark_crawling():
    urls = ["url1", "url2", ...] * 10  # 100+ URLs
    
    start_time = time.time()
    results = await crawl_site(urls, max_concurrent=5)
    end_time = time.time()
    
    print(f"Crawled {len(urls)} URLs in {end_time - start_time:.2f}s")
    print(f"Rate: {len(urls)/(end_time - start_time):.1f} URLs/second")
    print(f"Success rate: {results.success_rate:.1%}")
```

---

## Quick Reference

### Most Common Patterns

```python
# Simple single page
result = await scrape_page("https://example.com")
print(result.markdown_content)

# Batch processing
urls = ["url1", "url2", "url3"]
results = await crawl_site(urls)

# With error handling
for result in results.results:
    if result.success:
        process_content(result.markdown_content)
    else:
        handle_error(result.error)

# Synchronous usage
result = scrape_page_sync("https://example.com")
```

### Configuration Shortcuts

```python
# High performance
results = await crawl_site(urls, max_concurrent=10)

# With custom business context
context = {"domain": "Technology", "industry": "Software"}
result = await scrape_page(url, business_context=context)
```

---

**For more examples and advanced usage patterns, see `example_usage.py` and `test_full_capability.py`.**

**Need help? Contact: nyimbi@gmail.com**