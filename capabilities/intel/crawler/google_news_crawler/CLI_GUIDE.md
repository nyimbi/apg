# Google News Crawler CLI Guide

**Version**: 1.0.0  
**Date**: June 28, 2025  
**Status**: ✅ **COMPLETE**

## 🎯 Overview

The Google News Crawler CLI provides a powerful command-line interface for news intelligence gathering with integrated Crawlee content enhancement. Designed for researchers, journalists, and analysts monitoring conflict and security developments, particularly in the Horn of Africa region.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL (optional, mock DB available for testing)

### Install CLI
```bash
# Navigate to the Google News Crawler directory
cd /path/to/google_news_crawler

# Install in development mode (recommended)
pip install -e .

# Or install from setup file
python setup_cli.py install

# Verify installation
gnews-crawler --version
```

### Install with Crawlee Enhancement
```bash
# Install with Crawlee dependencies
pip install -e ".[crawlee]"

# Install Playwright browsers
playwright install chromium

# Verify Crawlee integration
gnews-crawler crawlee --status
```

## 📋 Commands Overview

| Command | Description | Example |
|---------|-------------|---------|
| `search` | Search for news articles | `gnews-crawler search "Ethiopia conflict"` |
| `monitor` | Continuous news monitoring | `gnews-crawler monitor --query "Horn of Africa"` |
| `config` | Configuration management | `gnews-crawler config --show` |
| `crawlee` | Crawlee integration tools | `gnews-crawler crawlee --test` |
| `status` | System status and health | `gnews-crawler status --check-deps` |

## 🔍 Search Command

Search for news articles with optional Crawlee enhancement.

### Basic Usage
```bash
# Simple search
gnews-crawler search "Ethiopia conflict"

# Search with specific countries
gnews-crawler search "Somalia crisis" --countries SO,ET,KE

# Search with Crawlee enhancement
gnews-crawler search "Sudan violence" --crawlee --max-results 20
```

### Advanced Options
```bash
# Time-filtered search
gnews-crawler search "displacement" --since 7d --until 2025-06-28

# Export results
gnews-crawler search "humanitarian crisis" --export results.json --format json

# Source filtering
gnews-crawler search "peacekeeping" --source-filter bbc.com,reuters.com

# Multiple languages
gnews-crawler search "الصراع" --languages ar,en --countries SD,EG
```

### Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Search query (supports boolean operators) |
| `--countries` | string | ET,SO,KE,SD,SS,UG,TZ | Comma-separated country codes |
| `--languages` | string | en,fr,ar | Comma-separated language codes |
| `--max-results` | int | 100 | Maximum number of results |
| `--crawlee` | flag | false | Enable Crawlee content enhancement |
| `--export` | string | none | Export file path (JSON, CSV, TXT) |
| `--format` | choice | table | Output format: table, json, csv, txt |
| `--source-filter` | string | none | Domain whitelist/blacklist |
| `--since` | string | none | Start date (YYYY-MM-DD or relative: 7d, 24h) |
| `--until` | string | none | End date (YYYY-MM-DD) |

### Output Formats

#### Table Format (Default)
```
📰 Articles (15 found):
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 1. Ethiopia PM announces new peace initiative in Tigray region... | BBC News        | ET | Q:0.85 W:1250
 2. Somalia receives humanitarian aid as drought conditions wors... | Reuters         | SO | Q:0.72 W:890
 3. Sudan conflict displaces thousands as fighting escalates...     | Al Jazeera      | SD | Q:0.68 W:1100
```

#### JSON Format
```json
[
  {
    "title": "Ethiopia PM announces new peace initiative",
    "url": "https://example.com/article",
    "publisher": {"name": "BBC News", "domain": "bbc.com"},
    "country": "ET",
    "published_date": "2025-06-28T10:30:00Z",
    "crawlee_enhanced": true,
    "full_content": "Complete article text...",
    "word_count": 1250,
    "crawlee_quality_score": 0.85,
    "geographic_entities": ["Ethiopia", "Tigray", "Addis Ababa"],
    "conflict_indicators": ["peace", "negotiation", "ceasefire"]
  }
]
```

## 📡 Monitor Command

Continuous monitoring of news with configurable intervals and alerts.

### Basic Monitoring
```bash
# Monitor every 5 minutes
gnews-crawler monitor --query "Horn of Africa crisis" --interval 300

# Monitor with Crawlee enhancement
gnews-crawler monitor --query "conflict" --crawlee --countries ET,SO

# Monitor with alerts
gnews-crawler monitor --query "security" --alert-keywords violence,attack,bombing
```

### Monitor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--query` | string | required | Search query to monitor |
| `--interval` | int | 300 | Monitoring interval in seconds |
| `--countries` | string | ET,SO,KE | Countries to monitor |
| `--crawlee` | flag | false | Enable Crawlee enhancement |
| `--max-results` | int | 50 | Max results per cycle |
| `--output-dir` | string | none | Directory to save results |
| `--alert-keywords` | string | none | Keywords triggering alerts |

### Monitoring Output
```
📊 Monitoring cycle 1 - 2025-06-28 14:30:00
   Found 12 total, 3 new articles
   💾 Saved 3 new articles to monitoring_cycle_1_20250628_143000.json

🚨 ALERT: 1 articles contain alert keywords!
   📰 Somalia security forces clash with militants in Mogadishu
      🔗 https://example.com/somalia-clash
```

## ⚙️ Config Command

Manage CLI configuration settings.

### Configuration Management
```bash
# Show current configuration
gnews-crawler config --show

# Initialize default configuration
gnews-crawler config --init

# Validate configuration
gnews-crawler config --validate

# Set configuration values
gnews-crawler config --set crawlee.max_requests 50
gnews-crawler config --set search.default_countries ET,SO,KE

# Get specific configuration
gnews-crawler config --get database.url
```

### Configuration Structure
```json
{
  "database": {
    "url": "postgresql://user:password@localhost:5432/gnews_db"
  },
  "crawlee": {
    "max_requests": 100,
    "max_concurrent": 5,
    "target_countries": ["ET", "SO", "KE", "UG", "TZ"],
    "enable_full_content": true,
    "min_content_length": 200,
    "enable_content_scoring": true
  },
  "search": {
    "default_countries": ["ET", "SO", "KE", "SD", "SS", "UG", "TZ"],
    "default_languages": ["en", "fr", "ar"],
    "default_max_results": 100
  },
  "monitoring": {
    "default_interval": 300,
    "default_output_dir": "~/gnews_monitoring"
  }
}
```

## 🕷️ Crawlee Command

Manage and test Crawlee integration.

### Crawlee Management
```bash
# Check Crawlee status
gnews-crawler crawlee --status

# Test Crawlee integration
gnews-crawler crawlee --test --max-requests 5

# Test specific extraction method
gnews-crawler crawlee --test --method trafilatura

# Generate configuration template
gnews-crawler crawlee --config-template --config crawlee_config.json
```

### Crawlee Status Output
```
🕷️ Crawlee Integration Status:
   Available: ✅ Yes
   Extraction methods available:
     trafilatura: ✅
     newspaper: ✅
     readability: ✅
     beautifulsoup: ✅
```

### Crawlee Test Output
```
🧪 Testing Crawlee integration...
📥 Testing content enhancement for 1 articles...
✅ Enhanced 1 articles

📄 BBC News Test Article
   URL: https://www.bbc.com/news/world-africa
   Success: True
   Method: trafilatura
   Word Count: 1240
   Quality Score: 0.87
   Geographic: ['Africa', 'Horn of Africa']
   Conflict: ['crisis', 'humanitarian']

📊 Processing Statistics:
   Total requests: 1
   Successful: 1
   Failed: 0
   Success rate: 100.0%
```

## 🔍 Status Command

Check system status and dependencies.

### Status Checks
```bash
# Basic status
gnews-crawler status

# Check all dependencies
gnews-crawler status --check-deps

# Test database connectivity
gnews-crawler status --test-db

# Test Crawlee integration
gnews-crawler status --test-crawlee
```

### Status Output
```
============================================================
Google News Crawler - System Status
============================================================

System Information:
  Python Version: 3.11.5
  Platform: darwin
  Current Time: 2025-06-28 14:30:00

📦 Dependency Check:

Core Dependencies:
   ✅ aiohttp: HTTP client for async operations
   ✅ asyncpg: PostgreSQL async driver
   ✅ feedparser: RSS/Atom feed parsing

Content Processing:
   ✅ beautifulsoup4: HTML parsing
   ✅ trafilatura: Content extraction
   ❌ newspaper3k: Article extraction
   ✅ readability-lxml: Content readability

Crawlee Integration:
   ✅ crawlee: Web crawling framework
   ✅ playwright: Browser automation

✅ Basic Status Check:
   Google News Crawler: Ready
   CLI Interface: Functional
   Core Dependencies: Available
```

## 🌍 Regional Focus: Horn of Africa

The CLI is optimized for Horn of Africa conflict monitoring:

### Default Countries
- **ET**: Ethiopia
- **SO**: Somalia  
- **KE**: Kenya
- **SD**: Sudan
- **SS**: South Sudan
- **UG**: Uganda
- **TZ**: Tanzania
- **ER**: Eritrea
- **DJ**: Djibouti

### Default Languages
- **en**: English
- **fr**: French
- **ar**: Arabic
- **sw**: Swahili

### Conflict Keywords
The CLI automatically enhances queries with conflict-related terms:
- conflict, violence, security, crisis, peace
- displacement, refugee, humanitarian
- attack, bombing, fighting, warfare

## 📊 Advanced Usage Examples

### 1. Comprehensive Regional Monitoring
```bash
# Monitor multiple conflict-related queries
gnews-crawler monitor \
  --query "Ethiopia OR Somalia OR Sudan conflict violence displacement" \
  --countries ET,SO,SD,SS,ER \
  --languages en,ar,fr \
  --crawlee \
  --interval 600 \
  --output-dir ./horn_africa_monitoring \
  --alert-keywords attack,bombing,violence,displacement
```

### 2. High-Quality Content Analysis
```bash
# Search with enhanced content extraction
gnews-crawler search "Horn of Africa humanitarian crisis" \
  --crawlee \
  --max-results 30 \
  --export analysis_$(date +%Y%m%d).json \
  --format json \
  --countries ET,SO,SD,SS,DJ,ER
```

### 3. Multi-Language News Research
```bash
# Search in multiple languages
gnews-crawler search "الأزمة الإنسانية" \
  --languages ar,en \
  --countries SD,EG,SA \
  --crawlee \
  --export arabic_news.csv \
  --format csv
```

### 4. Real-time Alert System
```bash
# Set up real-time monitoring with alerts
gnews-crawler monitor \
  --query "Ethiopia Tigray conflict ceasefire peace" \
  --interval 180 \
  --crawlee \
  --alert-keywords ceasefire,peace,agreement,negotiation \
  --output-dir ./ethiopia_peace_watch
```

## 🔧 Configuration Best Practices

### Development Configuration
```bash
gnews-crawler config --set crawlee.max_concurrent 3
gnews-crawler config --set crawlee.crawl_delay 2.0
gnews-crawler config --set search.default_max_results 20
```

### Production Configuration
```bash
gnews-crawler config --set crawlee.max_concurrent 8
gnews-crawler config --set crawlee.crawl_delay 1.0
gnews-crawler config --set crawlee.enable_caching true
```

### Research Configuration
```bash
gnews-crawler config --set crawlee.preferred_extraction_method trafilatura
gnews-crawler config --set crawlee.enable_content_scoring true
gnews-crawler config --set crawlee.min_content_length 500
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Command Not Found
```
gnews-crawler: command not found
```
**Solution**: Install the CLI package
```bash
pip install -e .
# or
python setup_cli.py install
```

#### 2. Crawlee Not Available
```
WARNING: Crawlee enhancement requested but crawlee_integration not available
```
**Solution**: Install Crawlee dependencies
```bash
pip install -e ".[crawlee]"
playwright install chromium
```

#### 3. Database Connection Errors
```
ERROR: Database connection failed
```
**Solution**: Use mock database for testing
```bash
# CLI automatically uses mock database when real database unavailable
gnews-crawler search "test query" --verbose
```

#### 4. Low Content Quality Scores
**Solutions**:
- Use different extraction methods: `--method trafilatura`
- Lower quality thresholds in config
- Check if content is behind paywalls

### Performance Optimization

#### For Speed
```bash
gnews-crawler config --set crawlee.max_concurrent 10
gnews-crawler config --set crawlee.crawl_delay 0.5
gnews-crawler config --set crawlee.enable_content_scoring false
```

#### For Quality
```bash
gnews-crawler config --set crawlee.max_concurrent 2
gnews-crawler config --set crawlee.crawl_delay 3.0
gnews-crawler config --set crawlee.preferred_extraction_method trafilatura
```

## 📈 Output Examples

### Search Results Table
```
📊 Search Results Analysis:
   Total articles: 25
   Top sources: BBC News(8), Reuters(6), Al Jazeera(4)
   Countries: ET(12), SO(8), SD(3), KE(2)
   Crawlee enhanced: 20/25 (80.0%)
   Average word count: 1150
   Average quality score: 0.78
   Average relevance score: 0.82

📰 Articles (25 found):
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 1. Ethiopia announces new humanitarian corridor for Tigray region assistance... | BBC News        | ET | Q:0.89 W:1340
 2. Somalia drought conditions worsen as international aid remains insufficient  | Reuters         | SO | Q:0.85 W:1180
 3. Sudan peace talks resume in Cairo with regional leaders participating...     | Al Jazeera      | SD | Q:0.82 W:1090
 4. Kenya deploys additional peacekeepers to South Sudan border region...        | Daily Nation    | KE | Q:0.78 W:920
```

### Monitoring Cycle
```
📡 Starting monitoring for: 'Horn of Africa security crisis'
🔄 Monitoring every 300 seconds. Press Ctrl+C to stop.

📊 Monitoring cycle 1 - 2025-06-28 14:30:00
   Found 18 total, 5 new articles
   💾 Saved 5 new articles to horn_africa_monitoring/monitoring_cycle_1_20250628_143000.json

📊 Monitoring cycle 2 - 2025-06-28 14:35:00
   Found 22 total, 2 new articles

🚨 ALERT: 1 articles contain alert keywords!
   📰 Ethiopia: Security forces clash with armed groups in Oromia region
      🔗 https://www.bbc.com/news/world-africa-ethiopia-clash-2025

📊 Monitoring cycle 3 - 2025-06-28 14:40:00
   Found 19 total, 1 new articles
```

## ✅ CLI Implementation Complete

The Google News Crawler CLI provides **comprehensive command-line access** to all crawler functionality with:

- **🔍 Powerful Search**: Multi-source news search with advanced filtering
- **📡 Continuous Monitoring**: Real-time news monitoring with alerts  
- **🕷️ Crawlee Integration**: Enhanced content extraction and analysis
- **⚙️ Configuration Management**: Flexible configuration system
- **🌍 Regional Focus**: Optimized for Horn of Africa conflict monitoring
- **📊 Multiple Output Formats**: Table, JSON, CSV, and text formats
- **🚨 Alert System**: Keyword-based alerting for critical events

The CLI is production-ready and provides enterprise-grade news intelligence capabilities through an intuitive command-line interface.