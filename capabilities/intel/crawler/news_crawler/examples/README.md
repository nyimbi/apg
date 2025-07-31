# News Crawler Examples

This directory contains comprehensive examples demonstrating how to use the Lindela News Crawler system with all its advanced features.

## 📋 Prerequisites

Before running the examples, ensure you have:

1. **Python 3.8+** installed
2. **Required dependencies** (see main package requirements)
3. **Optional: PostgreSQL database** (for database integration examples)
4. **Network connectivity** for web crawling

## 🚀 Quick Start

### Basic Usage

```python
from lindela.packages_enhanced.crawlers.news_crawler import ComprehensiveNewsCrawler

# Create a basic crawler
crawler = ComprehensiveNewsCrawler()

# Add news sites to crawl
crawler.add_target("https://example-news.com")
crawler.add_target("https://another-news-site.com")

# Perform comprehensive crawl
results = await crawler.crawl_all_targets()

print(f"Extracted {results.articles_extracted} articles")
```

### Enhanced Crawling with Database

```python
from lindela.packages_enhanced.crawlers.news_crawler import create_comprehensive_crawler

# Create configuration with database support
config = {
    'enable_database': True,
    'database_connection_string': 'postgresql://user:pass@localhost:5432/db',
    'enable_ml_analysis': True,
    'enable_monitoring': True
}

# Create enhanced crawler
crawler = await create_comprehensive_crawler(config)

# Crawl with automatic database storage
results = await crawler.crawl_all_targets()
```

### Stealth Crawling

```python
from lindela.packages_enhanced.crawlers.news_crawler import create_comprehensive_config

# Create stealth configuration
config = create_comprehensive_config(
    enable_stealth=True,
    enable_bypass=True,
    max_concurrent=2
)

crawler = await create_comprehensive_crawler(config)
```

## 📁 Example Files

### `comprehensive_example.py`

The main example file demonstrating all crawler features:

- **Basic Crawling**: Simple multi-site crawling
- **Stealth Crawling**: Anti-detection and bypass features
- **Database Integration**: Automatic article storage
- **Single URL Crawling**: Detailed single-page analysis
- **Performance Monitoring**: Metrics and performance tracking
- **Results Export**: Saving crawl results to files

**Run the examples:**

```bash
cd examples/
python comprehensive_example.py
```

## 🔧 Configuration Options

### Basic Configuration

```python
config = {
    'max_concurrent_targets': 3,        # Concurrent target sites
    'max_concurrent_urls': 5,           # Concurrent URLs per target
    'enable_stealth': False,            # Stealth mode
    'enable_bypass': True,              # Bypass protection
    'enable_database': False,           # Database storage
    'enable_caching': True,             # Result caching
}
```

### Stealth Configuration

```python
stealth_config = {
    'enable_stealth': True,
    'stealth_config': {
        'stealth_level': 'high',        # low, medium, high, maximum
        'randomize_user_agent': True,
        'randomize_headers': True,
        'enable_proxy_rotation': False,
        'min_delay': 2.0,
        'max_delay': 5.0
    }
}
```

### Database Configuration

```python
database_config = {
    'enable_database': True,
    'database_connection_string': 'postgresql://user:pass@host:port/db',
    'enable_ml_analysis': True,
    'database_config': {
        'pool_size': (10, 50),
        'timeout': 30
    }
}
```

### Parser Configuration

```python
parser_config = {
    'parser_config': {
        'enable_html_parsing': True,
        'enable_article_extraction': True,
        'enable_metadata_extraction': True,
        'enable_ml_analysis': True,
        'min_content_length': 100,
        'quality_threshold': 0.5
    }
}
```

## 📊 Features Demonstrated

### 1. Comprehensive Crawling
- Multi-target crawling with priority handling
- Concurrent processing with rate limiting
- Automatic URL discovery and following
- Content quality assessment

### 2. Content Extraction
- Multiple extraction strategies (newspaper3k, trafilatura, etc.)
- Metadata extraction (authors, dates, keywords)
- ML-powered content analysis
- Sentiment analysis and entity recognition

### 3. Stealth Capabilities
- Browser fingerprint spoofing
- User agent rotation
- Header randomization
- Rate limiting and throttling
- Session management

### 4. Bypass Systems
- Cloudflare bypass
- Anti-403 error handling
- JavaScript challenge solving
- CAPTCHA detection

### 5. Database Integration
- Automatic article storage
- ML analysis results persistence
- Event extraction and scoring
- Performance metrics tracking

### 6. Performance Monitoring
- Real-time metrics collection
- Response time tracking
- Success/failure rates
- Component health monitoring

## 🎯 Use Cases

### News Monitoring
```python
# Monitor specific news sources
targets = [
    CrawlTarget(url="https://bbc.com/news", name="BBC News", priority=1),
    CrawlTarget(url="https://reuters.com", name="Reuters", priority=1),
    CrawlTarget(url="https://ap.org", name="AP News", priority=2)
]

crawler.add_targets(targets)
results = await crawler.crawl_all_targets()
```

### Content Research
```python
# Research specific topics with ML analysis
config = create_comprehensive_config(enable_ml_analysis=True)
crawler = await create_comprehensive_crawler(config)

# Add topic-specific targets
crawler.add_target("https://example.com/tech-news")
results = await crawler.crawl_all_targets()

# Analyze sentiment and topics
for article in results.articles:
    if article.ml_analysis:
        print(f"Sentiment: {article.ml_analysis['sentiment_label']}")
        print(f"Topics: {article.ml_analysis['topics']}")
```

### Competitive Intelligence
```python
# Monitor competitor content with stealth
config = create_comprehensive_config(enable_stealth=True)
crawler = await create_comprehensive_crawler(config)

# Add competitor sites
competitors = ["competitor1.com", "competitor2.com"]
for site in competitors:
    crawler.add_target(f"https://{site}/news")

results = await crawler.crawl_all_targets()
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Check component availability
   from lindela.packages_enhanced.crawlers.news_crawler import get_crawler_health
   health = get_crawler_health()
   print(health)
   ```

2. **Database Connection Issues**
   ```python
   # Test database connection
   try:
       crawler = await create_comprehensive_crawler({
           'enable_database': True,
           'database_connection_string': 'postgresql://...'
       })
       print("Database connection successful")
   except Exception as e:
       print(f"Database error: {e}")
   ```

3. **Rate Limiting Issues**
   ```python
   # Adjust rate limiting
   config = {
       'max_concurrent_targets': 1,
       'max_concurrent_urls': 2,
       'crawler_config': {
           'request_delay': 3.0  # Increase delay
       }
   }
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
config = {'enable_monitoring': True}
crawler = await create_comprehensive_crawler(config)
```

## 📈 Performance Tips

1. **Optimize Concurrency**
   - Start with low concurrency (2-3)
   - Increase gradually based on target site limits
   - Monitor success rates

2. **Use Caching**
   - Enable caching for repeated crawls
   - Adjust TTL based on content freshness needs

3. **Database Optimization**
   - Use connection pooling
   - Batch article storage
   - Index frequently queried fields

4. **Stealth vs Performance**
   - Stealth mode reduces speed but increases success
   - Use bypass mode for protected sites
   - Balance stealth level with requirements

## 🆘 Support

For issues or questions:

1. Check the main package documentation
2. Review error logs and component availability
3. Test with simple examples first
4. Ensure all dependencies are installed

## 📄 License

MIT License - See main package for details.

---

**Happy Crawling! 🕷️**