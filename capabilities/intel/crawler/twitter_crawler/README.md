# Lindela Twitter Crawler Package

A comprehensive Twitter crawling and monitoring system using the twikit library for advanced social media intelligence and conflict monitoring. This package provides robust Twitter data collection, real-time monitoring, sentiment analysis, and conflict detection capabilities specifically designed for the Lindela spatial intelligence platform.

## 🌟 Features

### Core Functionality
- **Advanced Authentication**: Secure session management with automatic persistence
- **Intelligent Rate Limiting**: Adaptive rate limiting with automatic backoff
- **Robust Error Handling**: Comprehensive error handling with retry mechanisms
- **Session Persistence**: Automatic session saving and restoration

### Search & Collection
- **Advanced Search**: Complex query building with filters and operators
- **Conflict-Specific Templates**: Pre-built search templates for conflict monitoring
- **Real-time Collection**: Continuous data collection with configurable intervals
- **User Timeline Crawling**: Complete user profile and timeline data extraction

### Monitoring & Alerts
- **Real-time Monitoring**: Continuous monitoring of keywords, hashtags, and users
- **Conflict Detection**: Automated detection of conflict-related events
- **Alert System**: Multi-level alerting with customizable thresholds
- **Trend Analysis**: Real-time trend detection and analysis

### Analysis & Intelligence
- **Sentiment Analysis**: Advanced sentiment analysis using multiple algorithms
- **Conflict Analysis**: Specialized conflict relevance detection
- **Network Analysis**: Social network analysis and influence scoring
- **Entity Extraction**: Location, organization, and person extraction

### Data Management
- **Structured Storage**: SQLite, PostgreSQL, MongoDB support
- **Multiple Export Formats**: JSON, CSV, Excel, Parquet export
- **Data Processing**: Advanced data cleaning and normalization
- **Batch Operations**: Efficient batch processing capabilities

## 📦 Installation

### Dependencies

**Core Requirements:**
```bash
pip install twikit asyncio
```

**Advanced Features:**
```bash
# For analysis capabilities
pip install nltk textblob pandas numpy

# For network analysis
pip install networkx

# For database storage
pip install sqlalchemy

# For enhanced exports
pip install openpyxl pyarrow

# Complete installation
pip install twikit asyncio nltk textblob pandas numpy networkx sqlalchemy openpyxl pyarrow
```

### Setup

1. **Install the package dependencies**
2. **Download required NLTK data** (for sentiment analysis):
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Quick Start

### Basic Tweet Search
```python
import asyncio
from lindela.packages_enhanced.crawlers.twitter_crawler import quick_search

async def basic_search():
    # Quick search for tweets
    results = await quick_search(
        query="armed conflict Syria",
        max_results=100,
        language="en"
    )
    
    print(f"Found {len(results)} tweets")
    for tweet in results[:5]:
        print(f"- {tweet['text'][:100]}...")

# Run the search
asyncio.run(basic_search())
```

### Authenticated Crawler Setup
```python
import asyncio
from lindela.packages_enhanced.crawlers.twitter_crawler import TwitterCrawler, TwitterConfig

async def setup_crawler():
    # Configure crawler
    config = TwitterConfig(
        username="your_username",
        password="your_password",
        email="your_email@example.com",
        rate_limit_requests_per_minute=30,
        auto_save_session=True
    )
    
    # Initialize crawler
    crawler = TwitterCrawler(config)
    await crawler.initialize()
    
    # Search for tweets
    tweets = await crawler.search_tweets("conflict monitoring", count=50)
    
    print(f"Retrieved {len(tweets)} tweets")
    return tweets

# Run the setup
tweets = asyncio.run(setup_crawler())
```

### Conflict Monitoring
```python
import asyncio
from lindela.packages_enhanced.crawlers.twitter_crawler import create_conflict_monitor

async def monitor_conflicts():
    # Create conflict monitor
    monitor = await create_conflict_monitor(
        keywords=["armed conflict", "terrorism", "protests"],
        locations=["Syria", "Ukraine", "Somalia"],
        alert_threshold=10,
        check_interval_minutes=5
    )
    
    # Add alert callback
    def handle_alert(alert):
        print(f"🚨 {alert.level.value.upper()} ALERT: {alert.title}")
        print(f"📍 Location: {alert.location}")
        print(f"📊 Tweet count: {alert.tweet_count}")
        print(f"🔗 Keywords: {', '.join(alert.keywords_triggered)}")
    
    monitor.alert_system.add_alert_callback(AlertLevel.HIGH, handle_alert)
    monitor.alert_system.add_alert_callback(AlertLevel.CRITICAL, handle_alert)
    
    # Monitor will run continuously
    print("Monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(60)
            status = monitor.get_monitoring_status()
            print(f"Status: {status['status']}, Cached tweets: {status['cached_tweets']}")
    except KeyboardInterrupt:
        await monitor.stop_monitoring()

# Run monitoring
asyncio.run(monitor_conflicts())
```

### Advanced Search with Filters
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import (
    TwitterSearchEngine, SearchQuery, TweetFilter, DateRange, GeographicFilter
)
from datetime import datetime, timedelta

async def advanced_search():
    # Create search engine
    search_engine = TwitterSearchEngine()
    
    # Configure filters
    date_filter = DateRange(
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )
    
    geo_filter = GeographicFilter(
        latitude=33.5138,  # Damascus coordinates
        longitude=36.2765,
        radius="50km"
    )
    
    tweet_filter = TweetFilter(
        exclude_retweets=True,
        min_followers=1000,
        verified_users_only=False,
        languages=["en", "ar"],
        date_range=date_filter,
        geographic_filter=geo_filter
    )
    
    # Create search query
    query = SearchQuery(
        query="Syria conflict OR Syrian war",
        max_results=200,
        tweet_filter=tweet_filter,
        sort_by="recent"
    )
    
    # Execute search
    result = await search_engine.search(query)
    
    print(f"Found {len(result.tweets)} tweets")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    # Analyze hashtags
    hashtags = result.get_hashtags()
    print(f"Top hashtags: {hashtags[:10]}")
    
    return result

# Run advanced search
result = asyncio.run(advanced_search())
```

### Data Analysis
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import TwitterAnalyzer
from lindela.packages_enhanced.crawlers.twitter_crawler.data import TweetModel

async def analyze_tweets(raw_tweets):
    # Convert raw tweets to models
    tweet_models = []
    for raw_tweet in raw_tweets:
        tweet_model = TweetModel.from_raw_tweet(raw_tweet)
        tweet_models.append(tweet_model)
    
    # Create analyzer
    analyzer = TwitterAnalyzer()
    
    # Analyze tweets
    analyses = analyzer.analyze_tweet_batch(tweet_models, include_network=True)
    
    # Get summary
    summary = analyzer.get_analysis_summary(analyses)
    
    print("Analysis Summary:")
    print(f"Total tweets analyzed: {summary['total_tweets']}")
    
    if 'sentiment_summary' in summary:
        sentiment = summary['sentiment_summary']
        print(f"Average polarity: {sentiment.get('average_polarity', 0):.3f}")
        print(f"Positive ratio: {sentiment.get('positive_ratio', 0):.2%}")
        print(f"Negative ratio: {sentiment.get('negative_ratio', 0):.2%}")
    
    if 'conflict_summary' in summary:
        conflict = summary['conflict_summary']
        print(f"Conflict-related tweets: {conflict.get('total_conflict_related', 0)}")
        print(f"Conflict ratio: {conflict.get('conflict_ratio', 0):.2%}")
        
        if 'threat_level_distribution' in conflict:
            threat_dist = conflict['threat_level_distribution']
            print(f"Threat levels: {threat_dist}")
    
    return analyses

# Example usage with existing tweets
# analyses = asyncio.run(analyze_tweets(tweets))
```

### Data Export
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import (
    TwitterDataProcessor, ExportFormat
)

async def export_data():
    # Create data processor
    processor = TwitterDataProcessor()
    
    # Process raw tweets
    tweet_models = await processor.process_raw_tweets(raw_tweets, store=True)
    
    # Export to different formats
    json_file = await processor.export_recent_tweets(
        hours=24,
        format=ExportFormat.JSON,
        filename="conflict_tweets.json"
    )
    
    csv_file = await processor.export_recent_tweets(
        hours=24,
        format=ExportFormat.CSV,
        filename="conflict_tweets.csv"
    )
    
    excel_file = await processor.export_recent_tweets(
        hours=24,
        format=ExportFormat.EXCEL,
        filename="conflict_tweets.xlsx"
    )
    
    print(f"Exported to:")
    print(f"- JSON: {json_file}")
    print(f"- CSV: {csv_file}")
    print(f"- Excel: {excel_file}")
    
    # Get processing stats
    stats = processor.get_processing_stats()
    print(f"Processing stats: {stats}")

# Run export
# asyncio.run(export_data())
```

## 🔧 Configuration

### TwitterConfig Options
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import TwitterConfig

config = TwitterConfig(
    # Authentication
    username="your_username",
    password="your_password",
    email="your_email@example.com",
    
    # Session management
    session_file="twitter_session.pkl",
    auto_save_session=True,
    session_timeout=3600,  # 1 hour
    
    # Rate limiting
    rate_limit_requests_per_minute=30,
    rate_limit_requests_per_hour=1000,
    wait_on_rate_limit=True,
    
    # Retry configuration
    max_retries=3,
    backoff_factor=2.0,
    retry_on_status=[429, 500, 502, 503, 504],
    
    # Timeouts
    connect_timeout=30,
    read_timeout=60,
    
    # Proxy settings (optional)
    proxy="http://proxy-server:8080",
    proxy_auth=("username", "password"),
    
    # Logging
    log_level="INFO",
    log_requests=False
)
```

### MonitoringConfig Options
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import MonitoringConfig

config = MonitoringConfig(
    # Keywords and targets
    keywords=["armed conflict", "terrorism", "protests"],
    hashtags=["#Syria", "#Ukraine", "#conflict"],
    users_to_monitor=["@UN", "@Reuters", "@BBCBreaking"],
    locations=["Syria", "Ukraine", "Somalia"],
    languages=["en", "ar", "es"],
    
    # Alert thresholds
    alert_threshold=10,  # Number of tweets to trigger alert
    time_window_minutes=60,  # Time window for threshold
    critical_threshold=50,  # Critical alert threshold
    
    # Monitoring intervals
    check_interval_seconds=300,  # 5 minutes
    trend_analysis_interval_minutes=60,  # 1 hour
    
    # Alert settings
    alert_cooldown_minutes=30,
    max_alerts_per_hour=10,
    enable_escalation=True,
    
    # Data retention
    max_stored_tweets=10000,
    data_retention_hours=72,
    
    # Analysis features
    sentiment_analysis=True,
    network_analysis=False,
    geographic_clustering=True
)
```

## 📊 Analysis Capabilities

### Sentiment Analysis
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import quick_sentiment_analysis

# Analyze sentiment of a tweet
result = quick_sentiment_analysis("The situation is getting worse with more violence")

print(f"Polarity: {result.polarity:.3f}")  # -1.0 to 1.0
print(f"Subjectivity: {result.subjectivity:.3f}")  # 0.0 to 1.0
print(f"Category: {result.category.value}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Emotions: {result.emotions}")
```

### Conflict Analysis
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import quick_conflict_analysis

# Analyze conflict relevance
result = quick_conflict_analysis("Breaking: Armed clashes reported in Damascus")

print(f"Conflict-related: {result.is_conflict_related}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Category: {result.category.value}")
print(f"Urgency score: {result.urgency_score:.3f}")
print(f"Threat level: {result.threat_level}")
print(f"Keywords found: {result.keywords_found}")
print(f"Entities: {result.entities}")
```

### Network Analysis
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import NetworkAnalyzer

# Build and analyze network
analyzer = NetworkAnalyzer()
analyzer.build_network(tweet_models)

# Analyze user influence
influence = analyzer.analyze_user_influence("user_id_123")
if influence:
    print(f"User: {influence.username}")
    print(f"Influence score: {influence.influence_score:.3f}")
    print(f"Centrality scores: {influence.centrality_scores}")
    print(f"Connections: {len(influence.connections)}")

# Detect communities
communities = analyzer.detect_communities()
print(f"Found {len(set(communities.values()))} communities")
```

## 🔍 Search Templates

### Pre-built Conflict Search Templates
```python
from lindela.packages_enhanced.crawlers.twitter_crawler.search import ConflictSearchTemplates

# Armed conflict events
builder = ConflictSearchTemplates.armed_conflict_query("Syria")
query = builder.build()
print(f"Armed conflict query: {query}")

# Refugee crisis
builder = ConflictSearchTemplates.refugee_crisis_query("Somalia")
query = builder.build()
print(f"Refugee crisis query: {query}")

# Terrorism events
builder = ConflictSearchTemplates.terrorism_query("Paris")
query = builder.build()
print(f"Terrorism query: {query}")

# Protests and unrest
builder = ConflictSearchTemplates.protest_unrest_query("Hong Kong")
query = builder.build()
print(f"Protest query: {query}")
```

### Custom Query Building
```python
from lindela.packages_enhanced.crawlers.twitter_crawler.search import QueryBuilder, FilterOperator

# Build complex query
builder = QueryBuilder()
builder.add_keyword("Syria", FilterOperator.AND)
builder.add_keyword("conflict", FilterOperator.OR)
builder.add_keyword("war", FilterOperator.OR)
builder.add_hashtag("Syria")
builder.exclude_keyword("fake")
builder.add_language("en")
builder.add_geocode(33.5138, 36.2765, "100km")

query = builder.build()
print(f"Custom query: {query}")
```

## 💾 Data Storage

### Database Storage
```python
from lindela.packages_enhanced.crawlers.twitter_crawler.data import SQLiteStorage

# Initialize storage
storage = SQLiteStorage("conflict_tweets.db")

# Store tweets
await storage.store_tweets(tweet_models)

# Search stored tweets
results = await storage.search_tweets(
    query="Syria",
    start_date=datetime.now() - timedelta(days=7),
    limit=100
)

print(f"Found {len(results)} stored tweets")
```

### Export Options
```python
from lindela.packages_enhanced.crawlers.twitter_crawler.data import DataExporter, ExportFormat

exporter = DataExporter("exports/")

# Export to JSON
json_file = exporter.export_tweets(tweet_models, ExportFormat.JSON)

# Export to CSV
csv_file = exporter.export_tweets(tweet_models, ExportFormat.CSV)

# Export to Excel
excel_file = exporter.export_tweets(tweet_models, ExportFormat.EXCEL)

# Export to Parquet (for big data)
parquet_file = exporter.export_tweets(tweet_models, ExportFormat.PARQUET)
```

## 🚨 Real-time Monitoring

### Setting Up Monitoring
```python
import asyncio
from lindela.packages_enhanced.crawlers.twitter_crawler import ConflictMonitor, MonitoringConfig

async def setup_monitoring():
    # Configure monitoring
    config = MonitoringConfig(
        keywords=["armed conflict", "terrorism", "protests"],
        locations=["Syria", "Ukraine", "Somalia"],
        alert_threshold=10,
        check_interval_seconds=300  # 5 minutes
    )
    
    # Create monitor
    monitor = ConflictMonitor(config)
    
    # Set up alert handlers
    def handle_critical_alert(alert):
        print(f"🚨 CRITICAL ALERT: {alert.title}")
        # Send email, push notification, etc.
    
    def handle_high_alert(alert):
        print(f"⚠️  HIGH ALERT: {alert.title}")
        # Log to monitoring system
    
    monitor.alert_system.add_alert_callback(AlertLevel.CRITICAL, handle_critical_alert)
    monitor.alert_system.add_alert_callback(AlertLevel.HIGH, handle_high_alert)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    return monitor

# Run monitoring
monitor = asyncio.run(setup_monitoring())
```

### Alert Management
```python
# Get active alerts
active_alerts = monitor.alert_system.get_active_alerts()
for alert in active_alerts:
    print(f"Alert: {alert.title} ({alert.level.value})")

# Get recent alerts
recent_alerts = monitor.get_recent_alerts(hours=24)
print(f"Alerts in last 24 hours: {len(recent_alerts)}")

# Acknowledge alert
monitor.alert_system.acknowledge_alert("alert_id", "analyst_name")

# Get monitoring status
status = monitor.get_monitoring_status()
print(f"Status: {status}")
```

## 🔧 Advanced Usage

### Custom Analysis Pipeline
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import (
    TwitterCrawler, TwitterSearchEngine, TwitterAnalyzer, TwitterDataProcessor
)

class ConflictIntelligencePipeline:
    def __init__(self):
        self.crawler = TwitterCrawler(TwitterConfig())
        self.search_engine = TwitterSearchEngine(self.crawler)
        self.analyzer = TwitterAnalyzer()
        self.processor = TwitterDataProcessor()
    
    async def run_intelligence_cycle(self, query: str):
        # 1. Collection
        search_result = await self.search_engine.search_conflict_events(
            event_type="armed_conflict",
            location="Syria",
            max_results=500
        )
        
        # 2. Processing
        tweet_models = await self.processor.process_raw_tweets(
            search_result.tweets,
            store=True
        )
        
        # 3. Analysis
        analyses = self.analyzer.analyze_tweet_batch(tweet_models)
        summary = self.analyzer.get_analysis_summary(analyses)
        
        # 4. Reporting
        report = self.generate_intelligence_report(summary)
        
        return report
    
    def generate_intelligence_report(self, summary):
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "recommendations": self.generate_recommendations(summary)
        }
    
    def generate_recommendations(self, summary):
        recommendations = []
        
        if summary.get('conflict_summary', {}).get('total_conflict_related', 0) > 50:
            recommendations.append("High conflict activity detected - increase monitoring")
        
        sentiment = summary.get('sentiment_summary', {})
        if sentiment.get('negative_ratio', 0) > 0.7:
            recommendations.append("Predominantly negative sentiment - potential crisis")
        
        return recommendations

# Usage
pipeline = ConflictIntelligencePipeline()
report = asyncio.run(pipeline.run_intelligence_cycle("Syria conflict"))
```

### Integration with Lindela Mapping
```python
from lindela.packages_enhanced.utils.mapping import quick_conflict_map
from lindela.packages_enhanced.crawlers.twitter_crawler import TwitterSearchEngine

async def create_conflict_map_from_tweets():
    # Search for geolocated conflict tweets
    search_engine = TwitterSearchEngine()
    result = await search_engine.search_conflict_events(
        event_type="armed_conflict",
        location="Syria",
        max_results=200
    )
    
    # Extract geolocated events
    events = []
    for tweet in result.tweets:
        if tweet.get('coordinates'):
            events.append({
                'latitude': tweet['coordinates']['lat'],
                'longitude': tweet['coordinates']['lon'],
                'event_type': 'battle',
                'timestamp': tweet['created_at'],
                'severity': 2,  # Based on sentiment/urgency analysis
                'description': tweet['text'][:100],
                'location_name': tweet.get('place_name', 'Unknown')
            })
    
    # Generate conflict map
    map_html = await quick_conflict_map(events)
    
    return map_html

# Create integrated map
map_html = asyncio.run(create_conflict_map_from_tweets())
```

## 📈 Performance Optimization

### Batch Processing
```python
# Process large datasets efficiently
async def process_large_dataset(tweet_batches):
    processor = TwitterDataProcessor()
    
    for i, batch in enumerate(tweet_batches):
        print(f"Processing batch {i+1}/{len(tweet_batches)}")
        
        # Process in smaller chunks
        chunk_size = 100
        for j in range(0, len(batch), chunk_size):
            chunk = batch[j:j+chunk_size]
            await processor.process_raw_tweets(chunk, store=True)
        
        # Rate limiting between batches
        await asyncio.sleep(1)
```

### Memory Management
```python
# Configure for memory efficiency
config = TwitterConfig(
    max_retries=2,  # Reduce retries
    session_timeout=1800,  # 30 minutes
)

monitoring_config = MonitoringConfig(
    max_stored_tweets=5000,  # Reduce cache size
    data_retention_hours=24,  # Shorter retention
    check_interval_seconds=600  # Less frequent checks
)
```

## 🔒 Security Considerations

### Credential Management
```python
import os
from lindela.packages_enhanced.crawlers.twitter_crawler import TwitterConfig

# Use environment variables for credentials
config = TwitterConfig(
    username=os.getenv('TWITTER_USERNAME'),
    password=os.getenv('TWITTER_PASSWORD'),
    email=os.getenv('TWITTER_EMAIL'),
    session_file=os.getenv('SESSION_FILE', 'session.pkl')
)
```

### Rate Limit Compliance
```python
# Configure conservative rate limits
config = TwitterConfig(
    rate_limit_requests_per_minute=20,  # Below Twitter limits
    rate_limit_requests_per_hour=800,   # Conservative limit
    wait_on_rate_limit=True,            # Always wait
    rate_limit_buffer=0.2               # 20% buffer
)
```

## 🐛 Troubleshooting

### Common Issues

**Authentication Errors:**
```python
# Check credentials and session
crawler = TwitterCrawler(config)
try:
    await crawler.initialize()
    print("Authentication successful")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Clear session and retry
    crawler.session.clear_session()
```

**Rate Limiting:**
```python
# Monitor rate limits
status = crawler.rate_limiter.get_status()
print(f"Requests last minute: {status['requests_last_minute']}")
print(f"Can make request: {status['can_make_request']}")
print(f"Wait time: {status['estimated_wait_time']}s")
```

**Memory Issues:**
```python
# Configure for memory efficiency
config = MonitoringConfig(
    max_stored_tweets=1000,
    data_retention_hours=12
)
```

### Debugging

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Package-specific logging
from lindela.packages_enhanced.crawlers.twitter_crawler import setup_logging
setup_logging(level="DEBUG")
```

**Check Package Status:**
```python
from lindela.packages_enhanced.crawlers.twitter_crawler import get_package_info

info = get_package_info()
print(f"Package info: {info}")

# Check dependencies
print(f"twikit available: {info['dependencies']['twikit']}")
print(f"Core module: {info['modules']['core']}")
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Open an issue on the project repository
4. Contact the Lindela development team

---

**Lindela Twitter Crawler** - Advanced social media intelligence for conflict monitoring and spatial analysis.