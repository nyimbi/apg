# Search Crawler Package

**Advanced Multi-Engine Search Crawler for Conflict Monitoring and Intelligence Gathering**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/datacraft-co-ke/lindela)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A sophisticated search crawler system that orchestrates multiple search engines to provide comprehensive coverage for conflict monitoring, intelligence gathering, and news analysis. Specifically designed for Horn of Africa conflict monitoring with advanced keyword analysis and result ranking.

## 🌟 Features

### 🔍 **11 Search Engines Supported**
- **Google Search** - Market leader with advanced parsing
- **Bing Search** - Microsoft's search with rich metadata
- **DuckDuckGo** - Privacy-focused search
- **Yandex** - Russian search for international coverage
- **Baidu** - Chinese search with UTF-8 support
- **Yahoo Search** - Alternative Bing-powered results
- **Startpage** - Privacy-focused Google proxy
- **SearX** - Meta-search aggregating multiple engines
- **Brave Search** - Independent search index
- **Mojeek** - Independent crawler-based search
- **Swisscows** - Family-friendly Swiss search

### ⚡ **Core Capabilities**
- **Parallel Multi-Engine Search** - Execute searches across all engines simultaneously
- **Intelligent Result Ranking** - Advanced algorithms for relevance, freshness, and authority
- **Conflict-Focused Analysis** - Specialized scoring for conflict monitoring
- **Smart Deduplication** - TF-IDF-based similarity detection
- **Content Integration** - Seamless integration with news crawler for full-text analysis
- **Real-Time Alerts** - Automated high-priority conflict alerts
- **Comprehensive Analytics** - Engine performance tracking and keyword effectiveness analysis

### 🌍 **Horn of Africa Specialization**
- **Complete Regional Coverage** - 11 countries with detailed keyword databases
- **Conflict Type Detection** - Violence, terrorism, displacement, political instability
- **Location Intelligence** - Cities, regions, ethnic groups, conflict zones
- **Temporal Analysis** - Breaking news detection and trend analysis
- **Entity Extraction** - Key actors, organizations, and relationships

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Search Engines](#search-engines)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Conflict Monitoring](#conflict-monitoring)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- aiohttp
- beautifulsoup4
- scikit-learn
- numpy
- geopy (optional, for location analysis)
- spacy (optional, for NLP features)
- textblob (optional, for sentiment analysis)

### Install Dependencies

```bash
# Core dependencies
pip install aiohttp beautifulsoup4 scikit-learn numpy

# Optional NLP dependencies
pip install spacy textblob geopy
python -m spacy download en_core_web_sm

# Development dependencies
pip install pytest pytest-asyncio black flake8
```

### Package Installation

```python
# The search_crawler is part of the Lindela packages_enhanced system
from packages_enhanced.crawlers.search_crawler import SearchCrawler, ConflictSearchCrawler

# For Crawlee-enhanced functionality (requires Crawlee)
from packages_enhanced.crawlers.search_crawler import CrawleeEnhancedSearchCrawler
```

### Crawlee Integration

The search crawler now includes **Crawlee integration** for robust content downloading and parsing:

```bash
# Install Crawlee for enhanced content extraction
pip install crawlee

# Optional: Install additional content extractors
pip install trafilatura newspaper3k readability-lxml beautifulsoup4
```

## ⚡ Quick Start

### Basic Multi-Engine Search

```python
import asyncio
from search_crawler import SearchCrawler, SearchCrawlerConfig

async def basic_search():
    # Configure with multiple engines
    config = SearchCrawlerConfig(
        engines=['google', 'bing', 'duckduckgo', 'brave'],
        max_results_per_engine=10,
        total_max_results=30,
        parallel_searches=True
    )
    
    crawler = SearchCrawler(config)
    
    try:
        # Perform search
        results = await crawler.search(
            query="Somalia conflict news",
            max_results=20,
            download_content=True
        )
        
        # Process results
        for result in results[:5]:
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Engines: {', '.join(result.engines_found)}")
            print(f"Score: {result.combined_score:.3f}")
            print("-" * 50)
            
    finally:
        await crawler.close()

# Run the search
asyncio.run(basic_search())
```

### Conflict Monitoring

```python
import asyncio
from search_crawler import ConflictSearchCrawler, ConflictSearchConfig

async def monitor_conflicts():
    # Configure for conflict monitoring
    config = ConflictSearchConfig(
        engines=['google', 'bing', 'yandex', 'brave'],
        max_results_per_engine=15,
        min_relevance_score=0.7,
        enable_alerts=True,
        escalation_threshold=0.8
    )
    
    crawler = ConflictSearchCrawler(config)
    
    try:
        # Search for conflicts in Horn of Africa
        results = await crawler.search_conflicts(
            region='horn_of_africa',
            keywords=['Ethiopia', 'conflict', 'violence'],
            time_range='week',
            max_results=50
        )
        
        # Check for alerts
        alerts = [r for r in results if r.requires_alert]
        if alerts:
            print(f"🚨 {len(alerts)} HIGH PRIORITY ALERTS!")
            for alert in alerts[:3]:
                print(f"- {alert.title}")
                print(f"  Reasons: {', '.join(alert.alert_reasons)}")
        
        # Show top results
        print(f"\n📊 Found {len(results)} conflict-related results")
        for result in results[:5]:
            print(f"🔥 {result.title}")
            print(f"   Conflict Score: {result.conflict_score:.3f}")
            print(f"   Locations: {[loc['name'] for loc in result.locations_mentioned[:2]]}")
            
    finally:
        await crawler.close()

asyncio.run(monitor_conflicts())
```

### Crawlee-Enhanced Search with Full Content

```python
import asyncio
from search_crawler import create_crawlee_enhanced_search_crawler

async def enhanced_search_with_content():
    # Create Crawlee-enhanced crawler
    crawler = await create_crawlee_enhanced_search_crawler(
        engines=['google', 'bing', 'duckduckgo', 'brave'],
        enable_content_extraction=True,
        target_countries=['ET', 'SO', 'KE', 'SD']
    )
    
    try:
        # Perform search with full content extraction
        results = await crawler.search_with_content(
            query="Horn of Africa conflict humanitarian crisis",
            max_results=20,
            extract_content=True
        )
        
        # Analyze extracted content
        for result in results[:5]:
            print(f"📰 {result.title}")
            print(f"🔗 {result.url}")
            print(f"📊 Quality: {result.content_quality_score:.3f}")
            print(f"📝 Words: {result.word_count}")
            print(f"🕷️ Method: {result.extraction_method}")
            print(f"🌍 Locations: {result.geographic_entities}")
            print(f"⚔️ Conflicts: {result.conflict_indicators}")
            print(f"📄 Content Preview: {result.extracted_content[:200]}...")
            print("-" * 50)
            
    finally:
        await crawler.close()

asyncio.run(enhanced_search_with_content())
```

## 🔍 Search Engines

### Supported Engines

| Engine | Code | Coverage | Features | Rate Limits |
|--------|------|----------|----------|-------------|
| Google | `google` | Global | News, time filters, advanced parsing | Strict |
| Bing | `bing` | Global | News, images, metadata extraction | Moderate |
| DuckDuckGo | `duckduckgo` | Global | Privacy-focused, no tracking | Lenient |
| Yandex | `yandex` | Russia/CIS | International content, Cyrillic | Moderate |
| Baidu | `baidu` | China/Asia | Chinese content, UTF-8 support | Strict |
| Yahoo | `yahoo` | Global | Bing-powered, alternative results | Moderate |
| Startpage | `startpage` | Global | Google proxy, anonymous viewing | Lenient |
| SearX | `searx` | Global | Meta-search, multiple instances | Very Lenient |
| Brave | `brave` | Global | Independent index, privacy-focused | Reasonable |
| Mojeek | `mojeek` | Global | Independent crawler, UK-based | Very Lenient |
| Swisscows | `swisscows` | Global | Family-friendly, semantic search | Lenient |

### Engine Selection

```python
from search_crawler.engines import get_available_engines, create_engine

# List all available engines
engines = get_available_engines()
print(f"Available engines: {engines}")

# Create specific engine
google = create_engine('google')
results = await google.search("test query", max_results=10)

# Use engine registry
from search_crawler.engines import SEARCH_ENGINES
for name, engine_class in SEARCH_ENGINES.items():
    print(f"{name}: {engine_class.__name__}")
```

## ⚙️ Configuration

### SearchCrawlerConfig

```python
from search_crawler import SearchCrawlerConfig

config = SearchCrawlerConfig(
    # Engine configuration
    engines=['google', 'bing', 'duckduckgo'],
    engine_weights={
        'google': 1.0,
        'bing': 0.8,
        'duckduckgo': 0.9
    },
    
    # Search parameters
    max_results_per_engine=20,
    total_max_results=50,
    parallel_searches=True,
    timeout=30.0,
    
    # Ranking configuration
    ranking_strategy='hybrid',  # 'relevance', 'freshness', 'authority', 'hybrid'
    deduplication_threshold=0.85,
    
    # Content downloading
    download_content=True,
    parse_content=True,
    extract_metadata=True,
    
    # Caching and rate limiting
    enable_cache=True,
    cache_ttl=3600,
    min_delay_between_searches=1.0,
    max_concurrent_downloads=5,
    
    # Stealth options
    use_stealth=True,
    rotate_user_agents=True,
    use_proxies=False
)
```

### ConflictSearchConfig

```python
from search_crawler import ConflictSearchConfig

config = ConflictSearchConfig(
    # Inherit from SearchCrawlerConfig
    engines=['google', 'bing', 'yandex', 'brave'],
    
    # Conflict-specific parameters
    conflict_regions=['horn_of_africa', 'middle_east'],
    monitor_keywords=['violence', 'conflict', 'attack'],
    
    # Scoring weights
    location_weight=0.3,
    temporal_weight=0.25,
    keyword_weight=0.25,
    source_weight=0.2,
    
    # Filtering
    min_relevance_score=0.6,
    max_age_days=7,
    trusted_sources=[
        'reuters.com', 'bbc.com', 'aljazeera.com',
        'france24.com', 'dw.com', 'africanews.com'
    ],
    
    # Alert thresholds
    enable_alerts=True,
    escalation_threshold=0.8,
    critical_keywords=[
        'casualties', 'killed', 'explosion', 'attack'
    ],
    
    # Analysis options
    extract_entities=True,
    analyze_sentiment=True,
    detect_locations=True,
    track_developments=True
)
```

## 📖 Usage Examples

### 1. Multi-Engine Comparison

```python
async def compare_engines():
    engines = ['google', 'bing', 'duckduckgo', 'yandex']
    query = "Ethiopia Tigray conflict"
    
    engine_results = {}
    
    for engine in engines:
        config = SearchCrawlerConfig(engines=[engine])
        crawler = SearchCrawler(config)
        
        results = await crawler.search(query, max_results=10)
        engine_results[engine] = {
            'count': len(results),
            'avg_relevance': sum(r.relevance_score for r in results) / len(results)
        }
        
        await crawler.close()
    
    # Compare results
    for engine, stats in engine_results.items():
        print(f"{engine}: {stats['count']} results, {stats['avg_relevance']:.3f} relevance")
```

### 2. Keyword Analysis

```python
from search_crawler.keywords import KeywordAnalyzer, ConflictKeywordManager

async def analyze_keywords():
    # Get search results
    crawler = SearchCrawler()
    results = await crawler.search("Horn of Africa conflict", max_results=20)
    
    # Combine all text content
    all_text = " ".join([f"{r.title} {r.snippet}" for r in results])
    
    # Analyze keyword effectiveness
    keyword_manager = ConflictKeywordManager()
    analyzer = KeywordAnalyzer()
    
    conflict_keywords = keyword_manager.get_high_priority_keywords()
    analyses = analyzer.analyze_text(all_text, conflict_keywords)
    
    # Generate report
    report = analyzer.generate_keyword_report(analyses)
    print(report)
    
    # Get suggestions for new keywords
    suggestions = analyzer.suggest_new_keywords(analyses)
    print(f"Suggested keywords: {suggestions}")
```

### 3. Horn of Africa Monitoring

```python
from search_crawler.keywords import HornOfAfricaKeywords

async def monitor_hoa_conflicts():
    # Initialize keyword system
    hoa_keywords = HornOfAfricaKeywords()
    
    # Generate search queries for specific countries
    countries = ['somalia', 'ethiopia', 'sudan']
    queries = hoa_keywords.generate_search_queries(
        countries=countries,
        conflict_level='high',
        max_queries=20
    )
    
    # Search with conflict crawler
    crawler = ConflictSearchCrawler()
    all_results = []
    
    for query in queries[:10]:  # Limit for demo
        results = await crawler.search_conflicts(
            region='horn_of_africa',
            keywords=[query],
            max_results=5
        )
        all_results.extend(results)
    
    # Analyze location distribution
    locations = {}
    for result in all_results:
        for loc in result.locations_mentioned:
            name = loc['name']
            locations[name] = locations.get(name, 0) + 1
    
    print("Top mentioned locations:")
    for location, count in sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {location}: {count} mentions")
```

### 4. Real-Time Monitoring

```python
async def real_time_monitoring():
    config = ConflictSearchConfig(
        engines=['google', 'bing', 'brave'],
        enable_alerts=True,
        escalation_threshold=0.8
    )
    
    crawler = ConflictSearchCrawler(config)
    
    # Monitor for 1 hour with 5-minute intervals
    await crawler.monitor_region(
        region='horn_of_africa',
        interval=300,  # 5 minutes
        duration=3600  # 1 hour
    )
    
    # Check alerts
    if crawler.alert_queue:
        print(f"Generated {len(crawler.alert_queue)} alerts during monitoring")
        for alert in crawler.alert_queue:
            print(f"🚨 {alert.title}")
            print(f"   Score: {alert.conflict_score:.3f}")
            print(f"   Reasons: {', '.join(alert.alert_reasons)}")
```

### 5. Custom Ranking

```python
from search_crawler.ranking import ResultRanker, RankingStrategy

async def custom_ranking():
    # Search with basic crawler
    crawler = SearchCrawler()
    results = await crawler.search("conflict news", max_results=30)
    
    # Apply different ranking strategies
    strategies = ['relevance', 'freshness', 'authority', 'hybrid']
    
    for strategy in strategies:
        ranker = ResultRanker(strategy=strategy)
        ranked = ranker.rank(results, query="conflict news")
        
        print(f"\n{strategy.upper()} Ranking - Top 3:")
        for i, result in enumerate(ranked[:3], 1):
            score = result.metadata.get('ranking_score', 0)
            print(f"{i}. {result.title} (Score: {score:.3f})")
```

## 🌍 Conflict Monitoring

### Horn of Africa Coverage

The search crawler includes comprehensive coverage for Horn of Africa conflict monitoring:

#### Countries Covered
- **Somalia** - 76 cities, 25 regions, 14 ethnic groups, 27 conflict zones
- **Ethiopia** - 14 major cities, 12 regions, 20 ethnic groups, 26 conflict zones  
- **Eritrea** - 14 cities, 6 regions, 9 ethnic groups, 15 conflict zones
- **Djibouti** - 14 cities, 6 regions, 9 ethnic groups, 11 conflict zones
- **Sudan** - 14 cities, 18 regions, 18 ethnic groups, 24 conflict zones
- **South Sudan** - 14 cities, 10 regions, 18 ethnic groups, 26 conflict zones
- **Kenya** - 14 cities, 8 regions, 20 ethnic groups, 26 conflict zones
- **Uganda** - 14 cities, 18 regions, 20 ethnic groups, 27 conflict zones
- **Rwanda** - 14 cities, 5 regions, 3 ethnic groups, 21 conflict zones
- **Burundi** - 14 cities, 18 regions, 4 ethnic groups, 25 conflict zones
- **DRC** - 14 cities, 26 regions, 28 ethnic groups, 69 conflict zones

#### Conflict Types Detected
- **Violence & Armed Conflict** - Direct violence, battles, armed clashes
- **Terrorism & Extremism** - Terrorist attacks, extremist activities
- **Political Instability** - Coups, election violence, government crises
- **Displacement & Humanitarian** - Refugee movements, humanitarian crises
- **Resource Conflicts** - Land disputes, water conflicts, pastoral conflicts
- **Ethnic & Communal** - Ethnic tensions, communal violence

### Sample Conflict Monitoring Workflow

```python
from search_crawler.examples.horn_of_africa_conflict_demo import HornOfAfricaConflictMonitor

async def full_monitoring_example():
    # Initialize comprehensive monitor
    monitor = HornOfAfricaConflictMonitor("./results")
    
    try:
        # Monitor all Horn of Africa countries
        results = await monitor.monitor_conflicts(
            countries=['somalia', 'ethiopia', 'eritrea', 'sudan'],
            time_range='week',
            max_results=100
        )
        
        # Analyze keyword effectiveness
        keyword_analysis = await monitor.analyze_keyword_effectiveness()
        
        # Generate comprehensive report
        report = monitor.generate_conflict_report()
        
        print(f"Monitoring complete:")
        print(f"- {len(results)} results found")
        print(f"- {len(monitor.alerts)} alerts generated")
        print(f"- Report saved to: {report}")
        
    finally:
        await monitor.close()
```

## 📚 API Reference

### Core Classes

#### SearchCrawler
Main multi-engine search orchestrator.

```python
class SearchCrawler:
    def __init__(self, config: SearchCrawlerConfig)
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        engines: Optional[List[str]] = None,
        download_content: Optional[bool] = None,
        **kwargs
    ) -> List[EnhancedSearchResult]
    
    def get_stats(self) -> Dict[str, Any]
    def clear_cache(self)
    async def close(self)
```

#### ConflictSearchCrawler
Specialized crawler for conflict monitoring.

```python
class ConflictSearchCrawler(SearchCrawler):
    def __init__(self, config: ConflictSearchConfig)
    
    async def search_conflicts(
        self,
        region: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 50,
        time_range: Optional[str] = 'week',
        **kwargs
    ) -> List[ConflictSearchResult]
    
    async def monitor_region(
        self,
        region: str,
        interval: int = 300,
        duration: Optional[int] = None
    )
    
    def get_conflict_stats(self) -> Dict[str, Any]
```

#### EnhancedSearchResult
Enhanced search result with multi-engine metadata.

```python
@dataclass
class EnhancedSearchResult(SearchResult):
    # Multi-engine tracking
    engines_found: List[str]
    engine_ranks: Dict[str, int]
    
    # Content data
    content: Optional[str]
    parsed_content: Optional[Dict[str, Any]]
    download_time: Optional[float]
    
    # Enhanced scoring
    combined_score: float
    quality_score: float
    freshness_score: float
    authority_score: float
```

#### ConflictSearchResult
Specialized result for conflict analysis.

```python
@dataclass
class ConflictSearchResult(EnhancedSearchResult):
    # Conflict relevance
    conflict_score: float
    conflict_type: Optional[str]
    
    # Location analysis
    locations_mentioned: List[Dict[str, Any]]
    primary_location: Optional[Dict[str, Any]]
    
    # Temporal analysis
    event_date: Optional[datetime]
    is_breaking: bool
    
    # Entity extraction
    entities: Dict[str, List[str]]
    key_actors: List[str]
    
    # Sentiment analysis
    sentiment_score: float
    sentiment_label: str
    escalation_indicators: List[str]
    
    # Alert status
    requires_alert: bool
    alert_reasons: List[str]
```

### Keyword Management

#### ConflictKeywordManager
Manages conflict-related keywords.

```python
class ConflictKeywordManager:
    def get_keywords_by_category(self, category: str) -> List[str]
    def get_high_priority_keywords(self) -> List[str]
    def get_weighted_keywords(self, min_weight: float = 0.7) -> Dict[str, float]
    def generate_search_queries(self, location_keywords: List[str], max_queries: int = 30) -> List[SearchQuery]
    def score_text_relevance(self, text: str, boost_recent: bool = True) -> float
```

#### HornOfAfricaKeywords
Horn of Africa specific keyword management.

```python
class HornOfAfricaKeywords:
    def get_conflict_keywords(self, severity_level: str = 'all') -> List[str]
    def get_country_keywords(self, country: str) -> List[str]
    def get_all_location_keywords(self) -> List[str]
    def generate_search_queries(self, countries: List[str] = None, conflict_level: str = 'all', max_queries: int = 50) -> List[str]
    def score_keyword_relevance(self, text: str, country: str = None) -> float
    def extract_entities(self, text: str) -> Dict[str, List[str]]
```

### Result Ranking

#### ResultRanker
Advanced ranking system with multiple strategies.

```python
class ResultRanker:
    def __init__(self, strategy: str = 'hybrid', config: Optional[Dict[str, Any]] = None)
    def rank(self, results: List[Any], query: str = "") -> List[Any]
    def get_ranking_stats(self) -> Dict[str, Any]
    def reset_stats(self)

class ConflictRanker(ResultRanker):
    # Specialized ranking for conflict monitoring
    pass
```

## 📊 Performance

### Benchmark Results

Performance tests conducted on a typical search query across all engines:

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Engines** | 11 | All engines operational |
| **Parallel Search Time** | 8-15 seconds | Depends on engine response times |
| **Sequential Search Time** | 45-90 seconds | Sum of individual engine times |
| **Deduplication Accuracy** | 85-92% | TF-IDF threshold dependent |
| **Cache Hit Rate** | 75% | With 1-hour TTL |
| **Memory Usage** | 50-150 MB | Varies with result count |

### Optimization Tips

1. **Engine Selection**: Choose 4-6 engines for optimal speed/coverage balance
2. **Parallel Execution**: Always enable parallel searches for production
3. **Content Download**: Disable for faster searches when full content isn't needed
4. **Caching**: Enable caching for repeated queries
5. **Rate Limiting**: Adjust delays based on your use case and engine tolerance

### Scaling Considerations

- **Concurrent Searches**: Limit to 3-5 simultaneous search operations
- **Request Rate**: Respect individual engine rate limits
- **Memory Management**: Monitor memory usage with large result sets
- **Proxy Rotation**: Use proxies for high-volume operations
- **Instance Distribution**: Consider distributing across multiple instances for heavy loads

## 🛠️ Advanced Configuration

### Custom Search Engine

```python
from search_crawler.engines import BaseSearchEngine, SearchResult, SearchResponse

class CustomSearchEngine(BaseSearchEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self.base_url = "https://api.customsearch.com"
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        return f"{self.base_url}/search?q={query}&start={offset}"
    
    async def _parse_results(self, html: str, query: str) -> List[SearchResult]:
        # Custom parsing logic
        return []

# Register custom engine
from search_crawler.engines import SEARCH_ENGINES
SEARCH_ENGINES['custom'] = CustomSearchEngine
```

### Custom Ranking Strategy

```python
from search_crawler.ranking import ResultRanker

class CustomRanker(ResultRanker):
    def _calculate_score(self, result, query, all_results):
        # Custom scoring logic
        base_score = super()._calculate_score(result, query, all_results)
        
        # Add custom factors
        custom_boost = 0.0
        if 'urgent' in result.title.lower():
            custom_boost = 0.2
        
        base_score.total_score += custom_boost
        return base_score
```

### Integration with News Crawler

```python
# The search crawler is designed to integrate with the news_crawler
from packages_enhanced.crawlers.news_crawler import EnhancedNewsCrawler

config = SearchCrawlerConfig(
    download_content=True,  # Enable content downloading
    parse_content=True,     # Enable content parsing
    extract_metadata=True   # Enable metadata extraction
)

crawler = SearchCrawler(config)
results = await crawler.search("query", download_content=True)

# Results will include full content, parsed metadata, and extracted entities
for result in results:
    print(f"Content length: {len(result.content or '')}")
    print(f"Parsed metadata: {result.parsed_content}")
```

## 🔧 Troubleshooting

### Common Issues

#### Search Engine Failures
```python
# Check engine health
from search_crawler.engines import create_engine

engine = create_engine('google')
is_healthy = await engine.test_search()
print(f"Engine healthy: {is_healthy}")

# Get engine statistics
stats = engine.get_stats()
print(f"Success rate: {stats['success_rate']:.1%}")
```

#### Rate Limiting
```python
# Adjust rate limits in configuration
config = SearchCrawlerConfig(
    min_delay_between_searches=2.0,  # Increase delay
    max_concurrent_downloads=3       # Reduce concurrency
)
```

#### Memory Issues
```python
# Limit result sizes and enable cleanup
config = SearchCrawlerConfig(
    total_max_results=50,   # Reduce result count
    cache_ttl=1800,         # Shorter cache TTL
    download_content=False  # Disable content download
)

# Clear cache periodically
crawler.clear_cache()
```

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check component availability
from search_crawler import get_search_crawler_health

health = get_search_crawler_health()
print(f"System health: {health}")
```

### Error Handling

```python
async def robust_search():
    crawler = SearchCrawler()
    
    try:
        results = await crawler.search("query")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        
        # Fallback to single engine
        config = SearchCrawlerConfig(engines=['duckduckgo'])
        fallback_crawler = SearchCrawler(config)
        return await fallback_crawler.search("query")
    finally:
        await crawler.close()
```

## 🤝 Contributing

We welcome contributions to improve the search crawler! Here's how you can help:

### Adding New Search Engines

1. **Create Engine Class**: Inherit from `BaseSearchEngine`
2. **Implement Required Methods**: `_build_search_url()`, `_parse_results()`
3. **Add to Registry**: Update `engines/__init__.py`
4. **Write Tests**: Add comprehensive test coverage
5. **Update Documentation**: Add engine to README and docs

### Improving Ranking Algorithms

1. **Extend ResultRanker**: Add new ranking strategies
2. **Test Performance**: Benchmark against existing algorithms
3. **Document Strategy**: Explain the ranking methodology

### Enhancing Keyword Systems

1. **Expand Coverage**: Add new regions or conflict types
2. **Improve Analysis**: Enhance keyword effectiveness algorithms
3. **Add Languages**: Support for non-English keyword sets

### Development Setup

```bash
# Clone repository
git clone https://github.com/datacraft-co-ke/lindela.git
cd lindela/src/lindela/packages_enhanced/crawlers/search_crawler

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black .
flake8 .

# Run examples
python examples/basic_search_demo.py
python examples/horn_of_africa_conflict_demo.py
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all public methods
- Add comprehensive docstrings
- Write unit tests for new features
- Maintain backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Search Engine Providers** - For providing accessible search APIs
- **Open Source Libraries** - BeautifulSoup, aiohttp, scikit-learn, and others
- **Conflict Monitoring Community** - For insights into monitoring requirements
- **Horn of Africa Experts** - For domain knowledge and keyword validation

## 📞 Support

- **Documentation**: Full API documentation available in `/docs`
- **Examples**: Comprehensive examples in `/examples` directory  
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join our discussion forums for help and tips

---

**Built with ❤️ by [Datacraft](https://www.datacraft.co.ke) for comprehensive conflict monitoring and intelligence gathering.**