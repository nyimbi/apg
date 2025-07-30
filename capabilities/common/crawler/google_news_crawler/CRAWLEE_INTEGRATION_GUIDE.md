# Google News Crawler + Crawlee Integration Guide

**Date**: June 28, 2025  
**Version**: 1.0.0  
**Status**: ✅ **IMPLEMENTED**

## 🎯 Overview

The Google News crawler now supports **Crawlee integration** for enhanced content downloading. This integration allows you to:

- **Download full article content** from Google News search results
- **Extract structured content** using multiple parsing methods (Trafilatura, Newspaper3k, Readability, BeautifulSoup)
- **Enhance metadata** with author information, images, keywords, and geographic entities
- **Analyze content quality** with automated scoring and relevance assessment
- **Target specific regions** with Horn of Africa focused content filtering

## 🚀 Quick Start

### Basic Usage

```python
from google_news_crawler import create_crawlee_enhanced_gnews_client

# Create client with Crawlee integration
client = await create_crawlee_enhanced_gnews_client(
    db_manager=your_db_manager
)

# Search with content enhancement
articles = await client.search_news(
    query="Ethiopia conflict",
    enable_crawlee=True  # Enable full content downloading
)

# Access enhanced content
for article in articles:
    if article.get('crawlee_enhanced'):
        print(f"Title: {article['title']}")
        print(f"Full Content: {article['full_content'][:500]}...")
        print(f"Word Count: {article['word_count']}")
        print(f"Quality Score: {article['crawlee_quality_score']}")
```

### Advanced Configuration

```python
# Custom Crawlee configuration
crawlee_config = {
    'max_requests': 50,
    'max_concurrent': 5,
    'target_countries': ['ET', 'SO', 'KE', 'UG'],
    'enable_full_content': True,
    'min_content_length': 500,
    'enable_content_scoring': True,
    'preferred_extraction_method': 'trafilatura'
}

client = await create_crawlee_enhanced_gnews_client(
    db_manager=your_db_manager,
    crawlee_config=crawlee_config
)
```

## 📊 Enhanced Data Structure

When `enable_crawlee=True`, articles are enhanced with additional fields:

### Core Content Enhancement
```python
{
    'title': 'Original or enhanced title',
    'full_content': 'Complete article text',
    'word_count': 1250,
    'reading_time_minutes': 6.25,
    'crawlee_quality_score': 0.85,
    'crawlee_enhanced': True
}
```

### Content Structure
```python
{
    'article_text': 'Main article content',
    'lead_paragraph': 'Opening paragraph',
    'body_paragraphs': ['Paragraph 1', 'Paragraph 2', ...],
    'content_extraction_method': 'trafilatura'
}
```

### Enhanced Metadata
```python
{
    'images': [{'url': 'image_url', 'alt': 'description'}],
    'crawlee_authors': ['Author 1', 'Author 2'],
    'crawlee_keywords': ['keyword1', 'keyword2'],
    'tags': ['tag1', 'tag2']
}
```

### Geographic & Topical Analysis
```python
{
    'geographic_entities': ['Ethiopia', 'Addis Ababa', 'Horn of Africa'],
    'conflict_indicators': ['violence', 'displacement', 'crisis'],
    'crawlee_relevance_score': 0.75
}
```

### Processing Metadata
```python
{
    'content_processing_time_ms': 2500.0,
    'crawlee_success': True,
    'crawlee_fallback_used': False,
    'crawlee_errors': []
}
```

## 🔧 Configuration Options

### CrawleeConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_requests_per_crawl` | int | 100 | Maximum URLs to process |
| `max_concurrent` | int | 5 | Concurrent downloads |
| `enable_full_content` | bool | True | Enable content extraction |
| `preferred_extraction_method` | str | "auto" | trafilatura, newspaper, readability, bs4 |
| `min_content_length` | int | 200 | Minimum content length |
| `enable_content_scoring` | bool | True | Enable quality scoring |
| `target_countries` | List[str] | ['ET', 'SO', ...] | Target countries |
| `crawl_delay` | float | 1.0 | Delay between requests |
| `max_retries` | int | 3 | Maximum retry attempts |

### Content Quality Scoring

Quality scores are calculated based on:
- **Content length** (40% weight)
- **Title quality** (20% weight)  
- **Metadata presence** (40% weight: authors, dates, keywords, description)

Score ranges:
- **0.8-1.0**: High quality, comprehensive content
- **0.6-0.8**: Good quality, adequate content
- **0.4-0.6**: Fair quality, basic content
- **0.0-0.4**: Poor quality, minimal content

## 📈 Performance Considerations

### Extraction Method Performance

| Method | Speed | Quality | Dependencies |
|--------|-------|---------|--------------|
| **Trafilatura** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | `trafilatura` |
| **Newspaper3k** | ⭐⭐⭐ | ⭐⭐⭐⭐ | `newspaper3k` |
| **Readability** | ⭐⭐⭐⭐ | ⭐⭐⭐ | `readability-lxml` |
| **BeautifulSoup** | ⭐⭐⭐⭐⭐ | ⭐⭐ | `beautifulsoup4` |

### Recommended Settings

#### For Speed (High-Volume Processing)
```python
crawlee_config = {
    'max_concurrent': 10,
    'crawl_delay': 0.5,
    'preferred_extraction_method': 'beautifulsoup',
    'enable_content_scoring': False
}
```

#### For Quality (Research/Analysis)
```python
crawlee_config = {
    'max_concurrent': 3,
    'crawl_delay': 2.0,
    'preferred_extraction_method': 'trafilatura',
    'enable_content_scoring': True,
    'min_content_length': 500
}
```

#### For Horn of Africa Focus
```python
crawlee_config = {
    'target_countries': ['ET', 'SO', 'ER', 'DJ', 'KE', 'UG', 'SD', 'SS'],
    'target_languages': ['en', 'fr', 'ar', 'sw'],
    'enable_content_scoring': True
}
```

## 🛠️ Factory Functions

### `create_crawlee_enhanced_gnews_client()`
Creates a complete Google News client with Crawlee integration pre-configured.

```python
client = await create_crawlee_enhanced_gnews_client(
    db_manager=db_manager,
    stealth_orchestrator=None,
    crawlee_config={
        'max_requests': 50,
        'target_countries': ['ET', 'SO', 'KE']
    }
)
```

### `create_enhanced_gnews_client()`  
Standard factory with optional Crawlee enhancer.

```python
# Create Crawlee enhancer separately
enhancer = await create_crawlee_enhancer(config)

# Create client with enhancer
client = await create_enhanced_gnews_client(
    db_manager=db_manager,
    crawlee_enhancer=enhancer
)
```

### `create_basic_gnews_client()`
Basic client with optional Crawlee enhancement.

```python
client = await create_basic_gnews_client(
    db_manager=db_manager,
    crawlee_enhancer=enhancer  # Optional
)
```

## 📚 Usage Examples

### Example 1: Basic Enhancement

```python
async def basic_example():
    # Create client
    client = await create_crawlee_enhanced_gnews_client(db_manager)
    
    # Search with enhancement
    articles = await client.search_news(
        query="Somalia humanitarian crisis",
        enable_crawlee=True,
        max_results=20
    )
    
    # Process enhanced articles
    for article in articles:
        if article.get('crawlee_enhanced'):
            print(f"Enhanced: {article['title']}")
            print(f"Content: {len(article['full_content'])} chars")
        else:
            print(f"Basic: {article['title']}")
    
    await client.close()
```

### Example 2: Quality Filtering

```python
async def quality_filtered_example():
    # Configure for high quality
    config = {
        'min_content_length': 1000,
        'enable_content_scoring': True,
        'min_quality_score': 0.7
    }
    
    client = await create_crawlee_enhanced_gnews_client(
        db_manager=db_manager,
        crawlee_config=config
    )
    
    articles = await client.search_news(
        query="Ethiopia conflict analysis",
        enable_crawlee=True
    )
    
    # Filter by quality
    high_quality = [
        a for a in articles 
        if a.get('crawlee_quality_score', 0) >= 0.8
    ]
    
    print(f"Found {len(high_quality)} high-quality articles")
```

### Example 3: Geographic Analysis

```python
async def geographic_analysis_example():
    client = await create_crawlee_enhanced_gnews_client(db_manager)
    
    articles = await client.search_news(
        query="Horn of Africa security",
        countries=['ET', 'SO', 'KE'],
        enable_crawlee=True
    )
    
    # Analyze geographic mentions
    all_entities = []
    for article in articles:
        entities = article.get('geographic_entities', [])
        all_entities.extend(entities)
    
    # Count entity mentions
    from collections import Counter
    entity_counts = Counter(all_entities)
    
    print("Top geographic entities mentioned:")
    for entity, count in entity_counts.most_common(10):
        print(f"  {entity}: {count} mentions")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Crawlee Not Available
```
WARNING: Crawlee enhancement requested but crawlee_integration not available
```
**Solution**: Install Crawlee dependencies
```bash
pip install crawlee playwright
playwright install chromium
```

#### 2. Content Enhancement Failed
```
WARNING: Crawlee enhancement failed, continuing with basic results
```
**Solutions**:
- Check network connectivity
- Reduce `max_concurrent` setting
- Increase `crawl_delay`
- Enable fallback methods

#### 3. Low Quality Scores
**Solutions**:
- Lower `min_quality_score` threshold
- Try different extraction methods
- Check if content is behind paywalls
- Verify target URLs are accessible

### Performance Optimization

#### For High Volume
```python
# Optimize for speed
crawlee_config = {
    'max_concurrent': 8,
    'crawl_delay': 0.5,
    'request_handler_timeout': 30,
    'max_retries': 1,
    'enable_content_scoring': False
}
```

#### For Accuracy
```python
# Optimize for quality
crawlee_config = {
    'max_concurrent': 2,
    'crawl_delay': 3.0,
    'request_handler_timeout': 60,
    'max_retries': 3,
    'preferred_extraction_method': 'trafilatura'
}
```

## 📋 Migration Guide

### From Basic Google News Client

**Before:**
```python
client = await create_enhanced_gnews_client(db_manager)
articles = await client.search_news("query")
```

**After (with Crawlee):**
```python
client = await create_crawlee_enhanced_gnews_client(db_manager)
articles = await client.search_news("query", enable_crawlee=True)
```

### Key Changes
1. **New factory function**: `create_crawlee_enhanced_gnews_client()`
2. **New parameter**: `enable_crawlee=True` in `search_news()`
3. **Enhanced data**: Additional fields in article results
4. **Backward compatibility**: Existing code works without changes

## ✅ Integration Complete

The Google News crawler now provides **comprehensive content enhancement** through Crawlee integration while maintaining **full backward compatibility** with existing implementations.

**Next Steps:**
1. Install Crawlee dependencies
2. Update your client creation to use enhanced factory functions
3. Enable Crawlee enhancement in search calls
4. Leverage enhanced content data for improved analysis