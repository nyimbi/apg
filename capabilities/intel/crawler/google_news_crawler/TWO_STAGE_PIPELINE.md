# Google News Two-Stage Content Pipeline

## Overview

The Google News crawler now implements a sophisticated **two-stage pipeline** that combines discovery and content downloading for comprehensive news monitoring:

### Stage 1: Article Discovery (Google News Crawler)
- **Purpose**: Discover relevant news items based on search queries
- **Data Retrieved**: 
  - Article metadata (title, source, URL, published date)
  - Brief summaries/snippets 
  - Source credibility metrics
  - Geographic and topical classification
- **Performance**: Fast, lightweight discovery of many articles
- **Storage**: Metadata stored in `information_units` schema

### Stage 2: Full Content Download (Gen Crawler Integration)  
- **Purpose**: Download complete article content from discovered URLs
- **Data Retrieved**:
  - Full article text content
  - Complete metadata and structured data
  - Images, links, and media references
  - Content quality analysis
- **Performance**: Detailed content extraction with respectful crawling
- **Storage**: Full content enriches existing `information_units` records

## Architecture Benefits

### 🎯 **Efficient Resource Usage**
- Discover many articles quickly via Google News RSS feeds
- Download full content only for relevant/high-priority articles
- Avoid unnecessary bandwidth on low-quality sources

### 🛡️ **Respectful Crawling**
- Rate limiting at both stages prevents server overload
- Circuit breaker patterns handle failures gracefully
- Configurable delays respect website terms of service

### 📊 **Complete Data Coverage**
- Stage 1 provides broad coverage and trend detection
- Stage 2 provides detailed content for analysis
- Combined approach maximizes both breadth and depth

### 🔄 **Flexible Pipeline Control**
- Can disable Stage 2 for discovery-only operation
- Configurable batch sizes and processing rates
- Priority-based content downloading

## Usage Examples

### Basic Two-Stage Pipeline

```python
from google_news_crawler import create_horn_africa_content_pipeline

# Create complete pipeline
pipeline = create_horn_africa_content_pipeline("postgresql:///lnd")
await pipeline.initialize()

# Execute both stages
enriched_articles = await pipeline.discover_and_download(
    queries=["Ethiopia conflict", "Somalia security"],
    language='en',
    country='ET'
)

# Each article now has both discovery metadata AND full content
for article in enriched_articles:
    print(f"Title: {article.google_news_record.title}")
    print(f"Source: {article.google_news_record.source_name}")
    print(f"Full Content: {article.full_content[:200]}...")
    print(f"Word Count: {article.word_count}")
```

### Discovery-Only Mode

```python
from google_news_crawler import create_content_pipeline

# Create pipeline without content download
pipeline = create_content_pipeline(
    enable_content_download=False  # Stage 1 only
)

# Fast discovery without downloading full content
articles = await pipeline.discover_and_download(
    queries=["breaking news Horn of Africa"]
)
```

### Custom Configuration

```python
from google_news_crawler import ContentPipelineConfig, GoogleNewsContentPipeline

# Configure both stages
config = ContentPipelineConfig(
    # Stage 1: Discovery settings
    max_articles_per_query=50,
    
    # Stage 2: Download settings  
    download_full_content=True,
    gen_crawler_concurrent_downloads=3,
    batch_size=5,
    delay_between_batches=2.0
)

pipeline = GoogleNewsContentPipeline(config)
```

## Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Search Query  │───▶│  Google News API │───▶│  Article URLs   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Information     │◀───│   Gen Crawler    │◀───│   URL Queue     │
│ Units Database  │    │  (Full Content)  │    │   (Batched)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Performance Characteristics

### Stage 1: Discovery
- **Speed**: ~10-50 articles/second (RSS feeds)
- **Resource Usage**: Low bandwidth, minimal processing
- **Scalability**: Easily handles 1000s of queries

### Stage 2: Content Download
- **Speed**: ~2-5 articles/second (full content extraction)
- **Resource Usage**: Higher bandwidth, content processing
- **Scalability**: Batched processing with respectful delays

### Combined Pipeline
- **Throughput**: Discovery-limited (Stage 1 is bottleneck for volume)
- **Quality**: Content-limited (Stage 2 provides detailed analysis)
- **Efficiency**: Optimal balance of speed and completeness

## Configuration Best Practices

### For High-Volume Monitoring
```python
config = ContentPipelineConfig(
    max_articles_per_query=100,      # Cast wide net
    download_full_content=True,
    batch_size=10,                   # Larger batches
    delay_between_batches=1.0,       # Faster processing
    gen_crawler_concurrent_downloads=5
)
```

### For Detailed Analysis
```python
config = ContentPipelineConfig(
    max_articles_per_query=25,       # Focused discovery
    download_full_content=True,
    batch_size=3,                    # Smaller batches
    delay_between_batches=3.0,       # Respectful crawling
    enable_content_analysis=True,    # Deep analysis
    gen_crawler_concurrent_downloads=2
)
```

### For Development/Testing
```python
config = ContentPipelineConfig(
    max_articles_per_query=5,        # Minimal discovery
    download_full_content=False,     # Discovery only
    enable_content_analysis=False,   # Skip analysis
)
```

## Error Handling

The pipeline includes comprehensive error handling at both stages:

### Stage 1 Error Handling
- Rate limiting with exponential backoff
- Circuit breaker for Google News API failures
- Duplicate URL detection and filtering
- Query validation and sanitization

### Stage 2 Error Handling  
- Individual URL failure doesn't stop batch processing
- Retry logic for transient network failures
- Content quality validation
- Graceful degradation when gen_crawler unavailable

### Pipeline Error Recovery
- Failed articles are logged but don't stop processing
- Statistics track success/failure rates
- Health checks monitor component status
- Automatic fallback to discovery-only mode

## Monitoring and Statistics

The pipeline provides comprehensive monitoring:

```python
stats = await pipeline.get_pipeline_stats()

print(f"Articles discovered: {stats['pipeline_stats']['articles_discovered']}")
print(f"Articles downloaded: {stats['pipeline_stats']['articles_downloaded']}")
print(f"Download success rate: {stats['performance_metrics']['download_success_rate']:.1f}%")
print(f"Processing rate: {stats['performance_metrics']['processing_rate']:.1f} articles/sec")
```

## Database Integration

### Information Units Schema
Both stages integrate with the `information_units` database schema:

- **Stage 1**: Creates records with discovery metadata
- **Stage 2**: Enriches records with full content

### Content Deduplication
- URL-based deduplication prevents duplicate downloads
- Content hashing detects duplicate articles from different sources  
- Configurable deduplication strategies

## Future Enhancements

### Planned Improvements
1. **Intelligent Prioritization**: ML-based article importance scoring
2. **Adaptive Scheduling**: Dynamic adjustment based on content velocity
3. **Content Classification**: Automatic tagging and categorization
4. **Real-time Streaming**: WebSocket-based live updates
5. **Multi-source Integration**: Additional news APIs beyond Google News

### Integration Opportunities
1. **Event Extraction**: Automatic conflict event detection
2. **Sentiment Analysis**: Content mood and bias analysis  
3. **Entity Recognition**: Automatic person/place/organization tagging
4. **Trend Detection**: Emerging story identification
5. **Alert Systems**: Real-time notification for priority content

---

**Author**: Nyimbi Odero  
**Company**: Datacraft (www.datacraft.co.ke)  
**Version**: 1.0.0  
**Last Updated**: July 2025