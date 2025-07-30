# Gen Crawler - Next-Generation Web Crawler

Advanced web crawler built on Crawlee's AdaptivePlaywrightCrawler for comprehensive full-site crawling with modern asyncio architecture and intelligent adaptation.

## 🚀 Features

- **AdaptivePlaywrightCrawler**: Advanced Crawlee-based crawling engine that automatically switches between HTTP and browser-based crawling
- **Full Site Crawling**: Comprehensive site discovery and content extraction with `await context.enqueue_links()`
- **Intelligent Adaptation**: Automatically optimizes crawling strategy based on site characteristics and performance
- **Modern Architecture**: Built on asyncio and latest Python patterns
- **Content-Aware**: AI-powered content analysis and classification using multiple extraction methods
- **CLI Interface**: Comprehensive command-line interface with deep parameterization
- **Markdown Export**: Clean markdown file generation for crawled content
- **Database Integration**: Seamless integration with PostgreSQL databases
- **Performance Monitoring**: Real-time performance tracking and optimization

## 📦 Installation

```bash
# Install Crawlee with all dependencies
pip install 'crawlee[all]'

# Install optional dependencies for enhanced content extraction
pip install trafilatura newspaper3k readability-lxml beautifulsoup4

# Install database support (optional)
pip install psycopg2-binary sqlalchemy
```

## 🎯 Quick Start

### Basic Usage (Python API)

```python
import asyncio
from gen_crawler.core import GenCrawler

async def main():
    # Create crawler with default settings
    crawler = GenCrawler()
    await crawler.initialize()
    
    # Crawl a website
    result = await crawler.crawl_site("https://example.com")
    
    print(f"Crawled {result.total_pages} pages")
    print(f"Success rate: {result.success_rate:.1f}%")
    
    # Cleanup
    await crawler.cleanup()

asyncio.run(main())
```

### CLI Usage

```bash
# Basic site crawl with markdown export
gen-crawler crawl https://example.com --output ./results --format markdown

# Advanced crawl with custom settings
gen-crawler crawl https://news-site.com \
  --max-pages 500 --max-concurrent 3 --crawl-delay 2.0 \
  --include-patterns article,news,story --output ./news

# Multiple sites with conflict monitoring
gen-crawler crawl https://site1.com https://site2.com \
  --conflict-keywords war,violence,crisis --format json

# Create configuration file
gen-crawler config --create --template news --output ./news-config.json

# Export existing data to markdown
gen-crawler export ./crawl-results.json --format markdown --output ./markdown/
```

## 🛠️ CLI Commands

### `crawl` - Crawl websites

```bash
gen-crawler crawl <URLs> [OPTIONS]
```

**Key Options:**
- `--output, -o`: Output directory (default: ./crawl-results)
- `--format, -f`: Output format (markdown, json, csv, html)
- `--max-pages`: Maximum pages per site (default: 500)
- `--max-concurrent`: Maximum concurrent requests (default: 5)
- `--crawl-delay`: Delay between requests in seconds (default: 2.0)
- `--include-patterns`: Comma-separated patterns to include
- `--exclude-patterns`: Comma-separated patterns to exclude
- `--conflict-keywords`: Keywords for conflict monitoring
- `--config, -c`: Configuration file path
- `--save-raw-html`: Save raw HTML files

**Performance Options:**
- `--request-timeout`: Request timeout in seconds
- `--max-retries`: Maximum retries per request
- `--max-depth`: Maximum crawl depth
- `--memory-limit`: Memory limit in MB

**Content Filtering:**
- `--min-content-length`: Minimum content length
- `--max-content-length`: Maximum content length
- `--exclude-extensions`: File extensions to exclude

**Stealth Options:**
- `--user-agent`: Custom user agent
- `--random-user-agents`: Use random user agents
- `--proxy-list`: Path to proxy list file
- `--ignore-robots-txt`: Ignore robots.txt

### `config` - Configuration management

```bash
# Create new configuration
gen-crawler config --create --template news --output ./config.json

# Validate configuration
gen-crawler config --validate ./config.json
```

**Templates:**
- `basic`: Default settings
- `news`: News site optimized
- `research`: Research optimized
- `monitoring`: Monitoring optimized

### `export` - Export crawl data

```bash
gen-crawler export ./results.json --format markdown --output ./markdown/
```

**Options:**
- `--format`: Export format (markdown, html, csv, pdf)
- `--filter-quality`: Filter by minimum quality score
- `--filter-type`: Filter by content type
- `--organize-by`: Organization structure (site, date, type, quality)

### `analyze` - Analyze crawl data

```bash
gen-crawler analyze ./results.json --output ./analysis.json
```

## 📝 Markdown Export

The crawler can export clean, well-formatted markdown files:

```bash
gen-crawler crawl https://example.com --format markdown --output ./markdown/
```

**Markdown Features:**
- Clean content extraction using multiple methods
- Metadata inclusion (URL, authors, dates, keywords)
- Image references
- Related links
- Conflict analysis (if keywords provided)
- Organized directory structure
- Index file generation

**Example Output Structure:**
```
markdown/
├── INDEX.md
├── example_com/
│   ├── article_1.md
│   ├── news_story_2.md
│   └── report_3.md
└── styles.css (for HTML export)
```

## ⚙️ Configuration

### Configuration File Format

```json
{
  "performance": {
    "max_pages_per_site": 500,
    "max_concurrent": 5,
    "request_timeout": 30,
    "crawl_delay": 2.0,
    "max_depth": 10
  },
  "content_filters": {
    "min_content_length": 100,
    "include_patterns": ["article", "news", "story"],
    "exclude_patterns": ["tag", "archive", "login"],
    "exclude_extensions": [".pdf", ".doc", ".zip"]
  },
  "adaptive": {
    "enable_adaptive_crawling": true,
    "strategy_switching_threshold": 0.8,
    "performance_monitoring": true
  },
  "stealth": {
    "enable_stealth": true,
    "user_agent": "GenCrawler/1.0",
    "respect_robots_txt": true
  },
  "database": {
    "enable_database": false,
    "connection_string": "postgresql://user:pass@localhost/db"
  }
}
```

### Environment Variables

```bash
export GEN_CRAWLER_MAX_PAGES=1000
export GEN_CRAWLER_MAX_CONCURRENT=3
export GEN_CRAWLER_CRAWL_DELAY=1.5
export GEN_CRAWLER_USER_AGENT="Custom Agent"
export GEN_CRAWLER_DB_CONNECTION="postgresql://..."
```

## 🔍 Content Analysis

The crawler provides intelligent content analysis:

### Content Classification
- **Article**: News articles, blog posts, stories
- **Page**: Static pages, about pages, contact pages
- **Listing**: Category pages, tag pages, archives
- **Snippet**: Short content, excerpts

### Quality Scoring
- Content length and structure
- Metadata presence (authors, dates, keywords)
- Language detection and readability
- Image and link analysis

### Conflict Monitoring
```bash
gen-crawler crawl https://news-site.com \
  --conflict-keywords war,violence,crisis,protest,attack \
  --format json
```

Automatically identifies and flags content related to specified keywords.

## 🏗️ Architecture

### Core Components

```python
from gen_crawler.core import GenCrawler, AdaptiveCrawler
from gen_crawler.config import GenCrawlerConfig
from gen_crawler.parsers import GenContentParser
```

### Adaptive Strategy Management

```python
from gen_crawler.core import AdaptiveCrawler, CrawlStrategy

# Create adaptive manager
adaptive = AdaptiveCrawler()

# Get recommendation for a site
strategy = adaptive.recommend_strategy("https://example.com")

# Update performance metrics
adaptive.update_strategy_performance(url, strategy, success_rate, load_time)
```

### Content Parsing

```python
from gen_crawler.parsers import GenContentParser

parser = GenContentParser()
parsed = parser.parse_content(url, html_content)

print(f"Quality Score: {parsed.quality_score}")
print(f"Content Type: {parsed.content_type}")
print(f"Word Count: {parsed.word_count}")
```

## 📊 Performance Monitoring

The crawler provides comprehensive performance monitoring:

### Real-time Statistics
- Pages crawled per second
- Success/failure rates
- Memory usage
- Response times

### Site Profiling
- Strategy effectiveness
- Optimal settings per site
- Performance trends
- Error patterns

### Adaptive Optimization
- Automatic strategy switching
- Concurrency adjustment
- Timeout optimization
- Retry logic tuning

## 🔧 Advanced Usage

### Custom Configuration

```python
from gen_crawler.config import create_gen_config

# Create custom configuration
config = create_gen_config()
config.settings.performance.max_pages_per_site = 1000
config.settings.content_filters.include_patterns = ["research", "paper", "study"]

# Use with crawler
crawler = GenCrawler(config.get_crawler_config())
```

### Database Integration

```python
# Enable database storage
config.settings.database.enable_database = True
config.settings.database.connection_string = "postgresql://user:pass@localhost/db"

crawler = GenCrawler(config.get_crawler_config())
result = await crawler.crawl_site("https://example.com")
# Results automatically stored in database
```

### Conflict Monitoring Setup

```python
from gen_crawler.examples.news_site_crawling import monitor_breaking_news

# Monitor multiple news sites for breaking news
news_sites = [
    "https://news-site-1.com",
    "https://news-site-2.com"
]

await monitor_breaking_news(news_sites, check_interval=300)  # Check every 5 minutes
```

## 🚨 Error Handling

The crawler includes comprehensive error handling:

- **Network Errors**: Automatic retry with exponential backoff
- **Content Errors**: Graceful fallback to alternative extraction methods
- **Resource Limits**: Memory and timeout protection
- **Site Blocks**: Anti-detection and stealth features

## 📈 Performance Tips

1. **Optimize Concurrency**: Start with 3-5 concurrent requests, adjust based on site response
2. **Use Appropriate Delays**: 1-3 seconds between requests for most sites
3. **Filter Content**: Use include/exclude patterns to focus on relevant content
4. **Monitor Memory**: Set memory limits for large crawls
5. **Enable Adaptive Mode**: Let the crawler optimize settings automatically

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 👨‍💻 Author

**Nyimbi Odero**  
Datacraft (www.datacraft.co.ke)  
Email: nyimbi@datacraft.co.ke

---

*Gen Crawler - Intelligent web crawling for the modern era.*