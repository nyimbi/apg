# News Crawler User Guide

A comprehensive guide to using the Lindela News Crawler package for advanced news article extraction and site crawling.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Advanced Features](#advanced-features)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Quick Start

```python
import asyncio
from news_crawler import NewsCrawler, crawl_url

# Simple URL crawling
async def quick_example():
    article = await crawl_url("https://example-news.com/article")
    if article:
        print(f"Title: {article.title}")
        print(f"Content: {article.content[:200]}...")

asyncio.run(quick_example())
```

## Installation

The news crawler is part of the Lindela packages_enhanced system and requires several dependencies:

### Core Dependencies
```bash
pip install aiohttp cloudscraper newspaper3k trafilatura beautifulsoup4
pip install playwright  # For advanced browser automation
playwright install  # Install browser binaries
```

### Optional Dependencies
```bash
pip install psycopg2-binary  # For database integration
pip install ollama openai    # For ML analysis
```

## Basic Usage

### 1. Single URL Crawling

```python
import asyncio
from news_crawler import NewsCrawler

async def crawl_single_url():
    # Create crawler with default configuration
    crawler = NewsCrawler()
    
    try:
        article = await crawler.crawl_url("https://news-site.com/article")
        if article:
            print(f"Title: {article.title}")
            print(f"Authors: {article.authors}")
            print(f"Publish Date: {article.publish_date}")
            print(f"Content: {article.content}")
            print(f"Quality Score: {article.quality_score}")
    finally:
        await crawler.cleanup()

asyncio.run(crawl_single_url())
```

### 2. Multiple URL Crawling

```python
from news_crawler import crawl_multiple

async def crawl_multiple_urls():
    urls = [
        "https://news-site.com/article1",
        "https://news-site.com/article2",
        "https://news-site.com/article3"
    ]
    
    articles = await crawl_multiple(urls)
    
    for article in articles:
        print(f"Successfully crawled: {article.title}")

asyncio.run(crawl_multiple_urls())
```

### 3. Deep Site Crawling

```python
from news_crawler import CloudScraperStealthCrawler

async def crawl_entire_site():
    crawler = CloudScraperStealthCrawler()
    
    try:
        # Discover and crawl entire news site
        articles = await crawler.crawl_entire_site(
            "https://news-site.com", 
            max_articles=100
        )
        
        print(f"Crawled {len(articles)} articles from the site")
        
        for article in articles:
            print(f"- {article.title}")
            
    finally:
        await crawler.cleanup()

asyncio.run(crawl_entire_site())
```

## Configuration

### Using Default Configuration

```python
from news_crawler import get_default_config

config = get_default_config()
print(config)
# {
#     'max_concurrent_requests': 5,
#     'requests_per_second': 2.0,
#     'request_timeout': 30,
#     'max_retries': 3,
#     'enable_stealth': True,
#     'enable_bypass': True,
#     'enable_enhanced_stealth': True,
#     'enable_ml_analysis': True
# }
```

### Custom Configuration

```python
from news_crawler import NewsCrawler

# Custom configuration
config = {
    'max_concurrent_requests': 10,
    'requests_per_second': 5.0,
    'enable_stealth': True,
    'enable_bypass': True,
    'min_delay': 1.0,
    'max_delay': 3.0,
    'cloudflare_wait_time': 10.0
}

crawler = NewsCrawler(config)
```

### Using Unified Configuration System

```python
from news_crawler.config import create_news_crawler_config

async def use_unified_config():
    # Create unified configuration
    config = await create_news_crawler_config()
    
    crawler = NewsCrawler(config)
    # Use crawler...
```

### Environment Variables

```bash
# Set configuration via environment variables
export NEWS_CRAWLER_MAX_CONCURRENT=10
export NEWS_CRAWLER_TIMEOUT=45
export NEWS_CRAWLER_ENABLE_STEALTH=true
export NEWS_CRAWLER_RATE_LIMIT=3.0
```

## Core Components

### 1. NewsCrawler (Enhanced Crawler)

The primary crawler with advanced stealth capabilities:

```python
from news_crawler import NewsCrawler, StealthConfig

# Create stealth configuration
stealth_config = StealthConfig(
    enable_cloudflare_bypass=True,
    enable_fingerprint_protection=True,
    min_delay=2.0,
    max_delay=5.0
)

config = {
    'stealth_config': stealth_config,
    'enable_ml_analysis': True,
    'enable_database_storage': False
}

crawler = NewsCrawler(config)
```

**Features:**
- CloudScraper-based Cloudflare bypass
- Browser fingerprint protection
- User agent rotation
- Intelligent delays and rate limiting
- ML-powered content analysis

### 2. CloudScraperStealthCrawler (Deep Crawler)

For comprehensive site crawling:

```python
from news_crawler import CloudScraperStealthCrawler, CrawlTarget

# Create crawl target
target = CrawlTarget(
    url="https://news-site.com",
    name="Example News",
    max_articles=200,
    max_depth=3
)

crawler = CloudScraperStealthCrawler()
await crawler.crawl_target(target)
```

**Features:**
- Site discovery via sitemaps, RSS feeds, and link crawling
- Deep link following with configurable depth
- Batch processing with rate limiting
- Advanced URL filtering and prioritization
- Progress monitoring and session tracking

### 3. Content Parsers

Advanced content extraction:

```python
from news_crawler.parsers import ContentParser

parser = ContentParser({
    'enable_ml_analysis': True,
    'quality_threshold': 0.5,
    'min_content_length': 200
})

# Parse HTML content
result = await parser.parse(html_content, url)
if result.success:
    print(f"Title: {result.title}")
    print(f"Content: {result.content}")
    print(f"Quality: {result.overall_quality}")
```

### 4. Bypass Manager

Handle protected sites:

```python
from news_crawler.bypass import BypassManager, create_stealth_bypass_config

# Create stealth bypass configuration
bypass_config = create_stealth_bypass_config()

bypass_manager = BypassManager(bypass_config)
result = await bypass_manager.fetch_with_bypass(url)

if result.success:
    print(f"Bypassed protection using: {result.bypass_method}")
```

## Advanced Features

### 1. Stealth Crawling

```python
from news_crawler import create_stealth_crawler

# Create maximum stealth crawler
crawler = create_stealth_crawler(aggressive=True)

article = await crawler.crawl_url(protected_url)
```

### 2. Database Integration

```python
config = {
    'enable_database': True,
    'database_connection_string': 'postgresql://user:pass@localhost/news_db'
}

crawler = NewsCrawler(config)
# Articles will be automatically stored in database
```

### 3. ML Content Analysis

```python
config = {
    'enable_ml_analysis': True,
    'ml_config': {
        'confidence_threshold': 0.7,
        'enable_sentiment_analysis': True,
        'enable_entity_extraction': True
    }
}

crawler = NewsCrawler(config)
article = await crawler.crawl_url(url)

# Access ML analysis results
if article.ml_analysis:
    print(f"Sentiment: {article.ml_analysis['sentiment']}")
    print(f"Entities: {article.ml_analysis['entities']}")
```

### 4. Batch Processing

```python
from news_crawler import CloudScraperStealthCrawler

crawler = CloudScraperStealthCrawler()

# Add multiple targets
targets = [
    "https://site1.com",
    "https://site2.com", 
    "https://site3.com"
]

for target_url in targets:
    articles = await crawler.crawl_entire_site(target_url)
    print(f"Crawled {len(articles)} from {target_url}")
```

## Examples

### Example 1: News Monitoring System

```python
import asyncio
from datetime import datetime, timedelta
from news_crawler import NewsCrawler

class NewsMonitor:
    def __init__(self):
        self.crawler = NewsCrawler({
            'enable_stealth': True,
            'enable_ml_analysis': True
        })
        self.monitored_sites = [
            "https://news-site1.com",
            "https://news-site2.com"
        ]
    
    async def monitor_news(self):
        """Monitor news sites for new articles."""
        while True:
            for site in self.monitored_sites:
                try:
                    articles = await self.crawler.discover_site_articles(site)
                    
                    # Filter recent articles
                    recent_articles = []
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    
                    for article_url in articles[:20]:  # Check latest 20
                        article = await self.crawler.crawl_url(article_url)
                        if (article and article.publish_date and 
                            article.publish_date > cutoff_time):
                            recent_articles.append(article)
                    
                    print(f"Found {len(recent_articles)} recent articles from {site}")
                    
                except Exception as e:
                    print(f"Error monitoring {site}: {e}")
            
            # Wait before next check
            await asyncio.sleep(3600)  # 1 hour
    
    async def cleanup(self):
        await self.crawler.cleanup()

# Usage
monitor = NewsMonitor()
try:
    await monitor.monitor_news()
finally:
    await monitor.cleanup()
```

### Example 2: Research Article Collection

```python
import asyncio
from news_crawler import CloudScraperStealthCrawler

async def collect_research_articles(topic_sites, keywords):
    """Collect articles from research sites on specific topics."""
    crawler = CloudScraperStealthCrawler({
        'min_articles_per_site': 50,
        'enable_ml_analysis': True
    })
    
    all_articles = []
    
    try:
        for site in topic_sites:
            print(f"Crawling {site}...")
            articles = await crawler.crawl_entire_site(site, max_articles=100)
            
            # Filter by keywords
            relevant_articles = []
            for article in articles:
                content_lower = article.content.lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    relevant_articles.append(article)
            
            print(f"Found {len(relevant_articles)} relevant articles")
            all_articles.extend(relevant_articles)
        
        return all_articles
        
    finally:
        await crawler.cleanup()

# Usage
sites = ["https://research-site1.com", "https://research-site2.com"]
keywords = ["artificial intelligence", "machine learning", "deep learning"]

articles = await collect_research_articles(sites, keywords)
print(f"Collected {len(articles)} research articles")
```

## Troubleshooting

### Common Issues

1. **Cloudflare Protection**
   ```python
   # Enable enhanced Cloudflare bypass
   config = {
       'enable_cloudflare_bypass': True,
       'cloudflare_wait_time': 20.0,
       'max_retries': 5
   }
   ```

2. **Rate Limiting / 429 Errors**
   ```python
   # Reduce crawling speed
   config = {
       'requests_per_second': 0.5,
       'min_delay': 5.0,
       'max_delay': 10.0
   }
   ```

3. **403 Forbidden Errors**
   ```python
   # Enable anti-403 handling
   config = {
       'enable_bypass': True,
       'bypass_config': {
           'enable_403_handling': True,
           'rotate_user_agents_on_403': True
       }
   }
   ```

4. **Memory Issues**
   ```python
   # Reduce concurrent requests
   config = {
       'max_concurrent_requests': 2,
       'batch_size': 10
   }
   ```

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check crawler health
from news_crawler import get_crawler_health
health = get_crawler_health()
print(health)
```

### Performance Monitoring

```python
from news_crawler import NewsCrawler

crawler = NewsCrawler({'enable_monitoring': True})

# After crawling
if crawler.performance_monitor:
    stats = crawler.performance_monitor.get_summary()
    print(f"Average response time: {stats['avg_response_time']}ms")
    print(f"Success rate: {stats['success_rate']}%")
```

## Best Practices

### 1. Respectful Crawling
- Always respect `robots.txt`
- Use reasonable delays between requests
- Don't overwhelm servers with too many concurrent requests

```python
config = {
    'requests_per_second': 1.0,  # 1 request per second
    'max_concurrent_requests': 3,
    'respect_robots_txt': True
}
```

### 2. Error Handling
Always use proper error handling and cleanup:

```python
async def safe_crawling():
    crawler = NewsCrawler()
    try:
        article = await crawler.crawl_url(url)
        return article
    except Exception as e:
        logger.error(f"Crawling failed: {e}")
        return None
    finally:
        await crawler.cleanup()
```

### 3. Configuration Management
Use environment variables for sensitive configuration:

```python
import os

config = {
    'database_connection_string': os.getenv('NEWS_CRAWLER_DB_URL'),
    'enable_ml_analysis': os.getenv('ENABLE_ML', 'true').lower() == 'true'
}
```

### 4. Content Quality
Filter low-quality content:

```python
article = await crawler.crawl_url(url)
if article and article.quality_score > 0.7:
    # Process high-quality article
    process_article(article)
```

### 5. Monitoring and Alerting
Implement health checks:

```python
async def health_check():
    health = get_crawler_health()
    if health['status'] != 'healthy':
        send_alert(f"Crawler unhealthy: {health}")
```

## API Reference

For detailed API documentation, see:
- [Architecture Guide](architecture.md) - System architecture and design
- [Configuration Reference](../config/) - Detailed configuration options
- [Examples](../examples/) - Complete working examples

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the examples in the `examples/` directory
3. Check component health with `get_crawler_health()`
4. Enable debug logging for detailed error information