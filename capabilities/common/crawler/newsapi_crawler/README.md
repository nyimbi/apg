# NewsAPI Crawler Package

A comprehensive crawler for news APIs including [NewsAPI](https://newsapi.org/) and [NewsData.io](https://newsdata.io/) with advanced filtering, caching, and data processing capabilities. This package provides a robust solution for retrieving and processing news articles, with special focus on conflict monitoring in the Horn of Africa region.

## Features

- **Multi-Provider Support**: Support for NewsAPI.org and NewsData.io APIs
- **Credit Management**: Intelligent credit tracking and usage optimization for NewsData.io free plans
- **Stealth Integration**: Integration with news_crawler for full article content extraction
- **Comprehensive API Client**: Full support for all API endpoints with intelligent error handling
- **Advanced Caching**: Disk and memory-based caching to reduce API usage and improve performance
- **Rate Limiting**: Smart rate limiting to stay within API quotas
- **Content Enrichment**: NLP-based processing to extract entities, events, and keywords
- **Conflict Detection**: Specialized detection of conflict events in news articles
- **Flexible Configuration**: Comprehensive configuration options with smart defaults
- **Batch Processing**: Efficient batch processing of multiple queries
- **Async Support**: Modern async/await design for high-performance applications

## Installation

The NewsAPI Crawler is part of the Lindela project's `packages_enhanced` module. To use it:

1. Ensure you have the NewsAPI Python client installed:
   ```
   pip install newsapi-python
   ```

2. For full functionality, install these optional dependencies:
   ```
   pip install spacy newspaper3k
   python -m spacy download en_core_web_sm
   ```

3. Set your API keys as environment variables:
   ```bash
   # For NewsAPI.org
   export NEWSAPI_KEY="your-newsapi-key-here"
   
   # For NewsData.io
   export NEWSDATA_API_KEY="your-newsdata-key-here"
   ```

## Quick Start

### NewsAPI.org Example

```python
import asyncio
from packages_enhanced.crawlers.newsapi_crawler import create_advanced_client

async def main():
    # Create an advanced client with caching
    client = await create_advanced_client(
        cache_dir="./newsapi_cache",
        cache_ttl=3600  # 1 hour cache TTL
    )
    
    try:
        # Search for articles
        articles = await client.search_articles(
            query="Ethiopia conflict",
            language="en",
            sort_by="relevancy",
            max_results=50
        )
        
        # Print results
        print(f"Found {len(articles)} articles")
        for article in articles[:5]:  # Print first 5
            print(f"- {article['title']}")
            
    finally:
        # Always close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### NewsData.io Example with Priority Location Monitoring

```python
import asyncio
from packages_enhanced.crawlers.newsapi_crawler import create_newsdata_client

async def main():
    # Create NewsData.io client with credit tracking
    client = create_newsdata_client(
        cache_dir="./newsdata_cache",
        cache_ttl=3600
    )
    
    try:
        # Check credits before starting
        credit_status = client.get_credit_status()
        print(f"Credits available: {credit_status['remaining_credits']}")
        
        # Priority location search with hierarchical strategy
        # Searches: Aweil → Karamoja → Mandera → Assosa
        results = await client.search_priority_locations(
            locations=["aweil", "karamoja", "mandera", "assosa"],
            max_credits=10,  # Conservative for free plan
            include_stealth_download=True
        )
        
        # Print results by location
        for location, data in results.items():
            if location.startswith("_"):  # Skip summary
                continue
            
            print(f"\n{location.upper()}:")
            print(f"  Articles: {data['total_articles_found']}")
            print(f"  Search level: {data['final_level']}")
            print(f"  Credits used: {data['credits_used']}")
            
            # Show top article
            if data['articles']:
                article = data['articles'][0]
                print(f"  Top result: {article['title']}")
                if article.get('download_success'):
                    content_len = len(article.get('extracted_text', ''))
                    print(f"  Full content: {content_len} characters")
        
        # Quick alert scan for urgent incidents
        alerts = await client.alert_scan(max_credits=3)
        if alerts['alert_count'] > 0:
            print(f"\n⚠️  {alerts['alert_count']} urgent alerts found!")
        
        # Check remaining credits
        credit_status = client.get_credit_status()
        print(f"\nCredits remaining: {credit_status['remaining_credits']}")
            
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Single Location Monitoring

```python
# Monitor specific location for recent activity
results = await client.monitor_location(
    location="aweil",
    hours_back=24,
    max_credits=3
)

print(f"Found {results['articles_found']} recent articles for Aweil")
```

## Priority Locations & Hierarchical Search

The NewsData.io client is optimized for monitoring four priority locations using a hierarchical search strategy:

### Priority Locations

| Location | District/County | Region/Province | Country |
|----------|-----------------|-----------------|---------|
| **Aweil** | Aweil Center County | Northern Bahr el Ghazal | South Sudan |
| **Karamoja** | Karamoja sub-region | Northern Uganda | Uganda |
| **Mandera** | Mandera County | North Eastern Province | Kenya |
| **Assosa** | Assosa Zone | Benishangul-Gumuz | Ethiopia |

### Search Strategy

The crawler implements an intelligent hierarchical search that maximizes the value from your 200 credits:

1. **Specific Location Level**: `"Aweil" conflict` → `"Karamoja" violence`
2. **District Level**: `"Aweil Center County" attack` → `"Mandera County" raid`
3. **Regional Level**: `"Northern Bahr el Ghazal" killed` → `"Karamoja sub-region" clash`
4. **Country Level**: `"South Sudan" conflict` → `"Uganda" violence` (only if needed)

### Conflict Keywords

The system prioritizes these conflict-related terms:
- **High Priority**: `killed`, `attack`, `violence`, `raid`, `conflict`, `clash`
- **Medium Priority**: `fighting`, `assault`, `battle`, `militant`, `insurgent`
- **Contextual**: `displaced`, `refugees`, `humanitarian crisis`, `cattle rustling`

### Credit Optimization

```python
# Adaptive search based on available credits
if credits > 50:
    # Comprehensive monitoring (all locations)
    results = await client.search_priority_locations(max_credits=20)
elif credits > 20:
    # Focused monitoring (2 locations)
    results = await client.search_priority_locations(
        locations=["aweil", "mandera"], 
        max_credits=10
    )
else:
    # Alert scanning only
    alerts = await client.alert_scan(max_credits=5)
```

## Package Structure

- `api/`: API client implementation
  - `newsapi_client.py`: NewsAPI.org client classes
  - `newsdata_client.py`: NewsData.io client with credit management
  - `factory.py`: Factory functions for client creation
- `models/`: Data models
  - `article.py`: Classes for articles and collections
- `parsers/`: Content extraction and processing
  - `content_parser.py`: Extract and analyze article content
- `config/`: Configuration management
  - `configuration.py`: Configuration classes and utilities
- `utils/`: Helper utilities
  - `helpers.py`: Date handling, text processing, etc.
  - `location_optimizer.py`: Priority location search optimization
- `examples/`: Usage examples
  - `comprehensive_example.py`: Advanced usage example
  - `newsdata_example.py`: NewsData.io integration examples
  - `priority_locations_example.py`: Location-specific monitoring examples

## Advanced Usage

### Content Enrichment

```python
from packages_enhanced.crawlers.newsapi_crawler import create_advanced_client
from packages_enhanced.crawlers.newsapi_crawler.parsers import ArticleParser

async def enrich_articles():
    client = await create_advanced_client()
    parser = ArticleParser()
    
    articles = await client.search_articles("Ethiopia peace agreement")
    
    for article in articles:
        # Extract full text
        article_data = await parser.parse_article(article)
        
        # Print entities and locations
        print(f"Article: {article_data.get('title')}")
        print(f"Entities: {article_data.get('entities', [])}")
        print(f"Locations: {article_data.get('locations', [])}")
        print("-" * 50)
```

### Batch Processing

```python
from packages_enhanced.crawlers.newsapi_crawler import create_batch_client

async def process_multiple_queries():
    batch_client = await create_batch_client()
    
    # Define queries
    queries = [
        "Ethiopia conflict",
        "Somalia security",
        "Sudan peace agreement"
    ]
    
    # Process all queries
    results = await batch_client.process_queries(
        queries=queries,
        language="en",
        sort_by="relevancy"
    )
    
    # Print summary
    for query, articles in results.items():
        print(f"{query}: {len(articles)} articles")
```

### Conflict Event Detection

```python
from packages_enhanced.crawlers.newsapi_crawler.parsers import EventDetector

def detect_conflict_events(articles):
    detector = EventDetector()
    
    for article in articles:
        content = article.get("content") or article.get("description", "")
        events = detector.detect_events(content)
        
        if events:
            print(f"Detected {len(events)} events in article: {article['title']}")
            for event in events:
                print(f"  - {event['text']} (Score: {event.get('conflict_score', 0):.2f})")
```

## Configuration

The package can be configured in various ways:

```python
from packages_enhanced.crawlers.newsapi_crawler.config import create_config_with_defaults

# Create configuration with overrides
config = create_config_with_defaults(
    api_key="your-api-key",  # Override API key
    enable_caching=True,
    cache_dir="./custom_cache",
    default_language="en",
    region_focus=["Ethiopia", "Somalia", "Sudan"]
)

# Save configuration to file
config.save("newsapi_config.json")
```

## API Quotas and Rate Limiting

The NewsAPI has different rate limits based on your subscription plan:

- **Developer**: 100 requests per day
- **Standard**: 500 requests per day
- **Premium**: Custom limits

The `NewsAPIAdvancedClient` automatically handles rate limiting to stay within these quotas. You can check your current rate limit status:

```python
client = await create_advanced_client()
status = client.get_rate_limit_status()
print(f"Remaining requests: {status['remaining']}")
print(f"Reset in: {status['reset_in_seconds']} seconds")
```

## Horn of Africa Focus

This package includes special features for monitoring conflicts in the Horn of Africa region:

- Country-specific search templates
- Location extraction specialized for the region
- Conflict event detection tuned for regional context
- Comprehensive example focusing on regional conflicts

## Comprehensive Example

See `examples/comprehensive_example.py` for a full-featured example that demonstrates:
- Searching for conflict news in the Horn of Africa
- Article content extraction and enrichment
- Event detection and analysis
- Result saving and reporting

## Author

Nyimbi Odero  
Datacraft (www.datacraft.co.ke)

## License

MIT