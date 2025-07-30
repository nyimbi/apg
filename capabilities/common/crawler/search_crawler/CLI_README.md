# Search Crawler CLI

A comprehensive command-line interface for the Search Crawler package with intelligent defaults and deep parameterization capabilities.

## 🚀 Quick Start

### Installation

```bash
# Setup CLI with dependencies
python setup_cli.py

# Or setup without installing dependencies
python setup_cli.py --no-deps

# Create system-wide command (requires sudo)
sudo python setup_cli.py --symlink
```

### Basic Usage

```bash
# Simple search
./search_crawler_cli "Ethiopia conflict"

# Conflict monitoring
./search_crawler_cli "Somalia violence" --mode conflict --enable-alerts

# Content extraction
./search_crawler_cli "Horn of Africa news" --mode crawlee --extract-content

# Interactive mode
./search_crawler_cli --interactive
```

## 📋 Features

### 🎯 **Intelligent Defaults**
- **Auto Mode**: Automatically selects optimal engines and settings based on query content
- **Smart Engine Selection**: Chooses best engines for each use case
- **Adaptive Performance**: Optimizes concurrency and timeouts based on mode

### 🔍 **Three Specialized Modes**

#### General Mode
- Balanced web search across multiple engines
- Optimized for general information retrieval
- Fast and reliable performance

#### Conflict Mode
- Horn of Africa conflict monitoring specialization
- Enhanced keyword detection for violence, terrorism, political instability
- Geographic entity extraction for 8+ countries
- Real-time alert system for high-priority events

#### Crawlee Mode
- Advanced content extraction using multiple methods
- Quality scoring and content analysis
- Full-text processing with 4 extraction engines
- Content filtering and relevance ranking

### ⚙️ **Deep Parameterization**

| Category | Parameters | Description |
|----------|------------|-------------|
| **Search** | engines, max-results, query | Core search configuration |
| **Content** | extract-content, min-content-length, preferred-extraction | Content processing |
| **Geographic** | target-countries, conflict-regions | Geographic targeting |
| **Performance** | timeout, max-concurrent, retry-attempts | Performance tuning |
| **Quality** | min-relevance-score, quality-filtering | Result filtering |
| **Output** | format, file, verbose | Output control |

### 📊 **Multiple Output Formats**
- **Table**: Rich formatted terminal output
- **JSON**: Structured data with metadata
- **CSV**: Spreadsheet-compatible format
- **YAML**: Human-readable configuration
- **Text**: Plain text for compatibility

## 🛠️ Command Reference

### Core Commands

```bash
# Basic search with intelligent defaults
./search_crawler_cli "your search query"

# Specify search mode
./search_crawler_cli "query" --mode {general|conflict|crawlee|auto}

# Select specific engines
./search_crawler_cli "query" --engines google,bing,duckduckgo

# Control result count
./search_crawler_cli "query" --max-results 50 --max-results-per-engine 15
```

### Conflict Monitoring

```bash
# Enable conflict detection and alerts
./search_crawler_cli "Ethiopia violence" --mode conflict --enable-alerts

# Target specific countries
./search_crawler_cli "refugee crisis" --target-countries ET,SO,SS,SD

# Set escalation threshold
./search_crawler_cli "Sudan conflict" --escalation-threshold 0.8
```

### Content Extraction

```bash
# Extract full content from results
./search_crawler_cli "news query" --mode crawlee --extract-content

# Choose extraction method
./search_crawler_cli "content" --preferred-extraction trafilatura

# Filter by content quality
./search_crawler_cli "articles" --min-quality-score 0.8
```

### Output Control

```bash
# Save to JSON file
./search_crawler_cli "query" --output-format json --output-file results.json

# Verbose logging
./search_crawler_cli "query" --verbose

# Quiet mode
./search_crawler_cli "query" --quiet
```

### Advanced Features

```bash
# Load configuration from file
./search_crawler_cli --load-config config.yaml

# Save current configuration
./search_crawler_cli "query" --save-config my_settings.yaml

# Continuous monitoring
./search_crawler_cli "monitor query" --monitor-mode --monitor-interval 300

# Performance benchmark
./search_crawler_cli --benchmark

# System health check
./search_crawler_cli --health-check
```

## 📁 Configuration Files

### Example Configurations

The `configs/` directory contains ready-to-use configurations:

- `example_general_search.yaml` - Basic web search
- `example_conflict_monitoring.yaml` - Security monitoring 
- `example_crawlee_enhanced.yaml` - Content extraction
- `example_monitoring_batch.yaml` - Batch processing

### Custom Configuration

```yaml
# my_config.yaml
mode: "conflict"
query: "Horn of Africa crisis"
engines:
  - "google"
  - "bing" 
  - "duckduckgo"
max_results: 50
target_countries:
  - "ET"
  - "SO"
  - "ER"
enable_alerts: true
output_format: "json"
```

Load with:
```bash
./search_crawler_cli --load-config my_config.yaml
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- Internet connection for search engines

### Dependencies

**Core (Required):**
```bash
pip install aiohttp beautifulsoup4 lxml requests httpx
```

**Enhanced CLI (Recommended):**
```bash
pip install rich click pyyaml
```

**Content Extraction (Optional):**
```bash
pip install trafilatura newspaper3k readability-lxml
```

### Setup Methods

#### Automatic Setup
```bash
# Complete setup with all features
python setup_cli.py

# Basic setup only
python setup_cli.py --basic

# System-wide installation
sudo python setup_cli.py --symlink
```

#### Manual Setup
```bash
# Make CLI executable
chmod +x cli.py
chmod +x search_crawler_cli

# Test installation
python cli.py --health-check
```

## 🎯 Use Cases

### 1. **Security Monitoring**
Monitor Horn of Africa conflicts in real-time:

```bash
./search_crawler_cli "Ethiopia Tigray conflict" \
  --mode conflict \
  --target-countries ET,ER \
  --enable-alerts \
  --monitor-mode \
  --output-format json \
  --output-file alerts.json
```

### 2. **Research & Analysis**
Extract high-quality content for analysis:

```bash
./search_crawler_cli "humanitarian aid effectiveness" \
  --mode crawlee \
  --extract-content \
  --min-quality-score 0.8 \
  --preferred-extraction trafilatura \
  --output-format json
```

### 3. **News Aggregation**
Collect news from multiple sources:

```bash
./search_crawler_cli "Somalia Al-Shabaab" \
  --engines google,bing,duckduckgo,yandex \
  --max-results 100 \
  --deduplicate-results \
  --output-format csv \
  --output-file news_aggregation.csv
```

### 4. **Trend Monitoring**
Track emerging topics and trends:

```bash
./search_crawler_cli "climate change Horn of Africa" \
  --monitor-mode \
  --monitor-interval 600 \
  --min-relevance-score 0.7 \
  --save-config trend_monitoring.yaml
```

## 🛡️ Best Practices

### Performance Optimization

1. **Engine Selection**: Use 3-5 engines for optimal speed/coverage
2. **Concurrency**: Start with defaults, adjust based on performance
3. **Timeouts**: Increase for content extraction, decrease for speed
4. **Rate Limiting**: Respect service limits with appropriate delays

### Quality Control

1. **Relevance Filtering**: Set appropriate minimum scores
2. **Content Length**: Filter short/low-quality content
3. **Deduplication**: Always enable for cleaner results
4. **Quality Scoring**: Use for content-focused searches

### Operational Security

1. **Stealth Mode**: Enabled by default for respectful crawling
2. **Rate Limits**: Built-in delays prevent service overload
3. **Error Handling**: Robust retry mechanisms included
4. **Privacy**: No personal data collection

## 🚨 Troubleshooting

### Common Issues

**No Results Found:**
```bash
# Try broader terms and lower thresholds
./search_crawler_cli "broader query" --min-relevance-score 0.3
```

**Timeout Errors:**
```bash
# Increase timeout and reduce concurrency
./search_crawler_cli "query" --timeout 60 --max-concurrent 5
```

**Import Errors:**
```bash
# Check component health
./search_crawler_cli --health-check

# Reinstall dependencies
python setup_cli.py
```

**Rate Limiting:**
```bash
# Increase delays between requests
./search_crawler_cli "query" --rate-limit-delay 3.0
```

### Debug Mode

```bash
# Enable verbose logging
./search_crawler_cli "query" --verbose

# Check system status
./search_crawler_cli --health-check

# Run basic benchmark
./search_crawler_cli --benchmark
```

## 📈 Performance Metrics

### Typical Performance

| Operation | Time | Results | Notes |
|-----------|------|---------|-------|
| Basic Search | 5-15s | 20-50 | 3-5 engines |
| Conflict Monitoring | 10-30s | 50-100 | Enhanced analysis |
| Content Extraction | 30-120s | 10-30 | Full text processing |

### Optimization Tips

- Use fewer engines for speed
- Reduce max results for faster searches
- Enable quality filtering only when needed
- Adjust concurrency based on system resources

## 🤝 Integration

### Shell Scripts

```bash
#!/bin/bash
# automated_monitoring.sh
./search_crawler_cli "$1" \
  --mode conflict \
  --enable-alerts \
  --output-format json \
  --output-file "alert_$(date +%Y%m%d_%H%M%S).json"
```

### Python Integration

```python
import subprocess
import json

def search_conflicts(query):
    result = subprocess.run([
        './search_crawler_cli', query,
        '--mode', 'conflict',
        '--output-format', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None
```

### API Endpoints

```bash
# Generate JSON for API consumption
./search_crawler_cli "API query" --output-format json | \
  curl -X POST -H "Content-Type: application/json" \
  -d @- https://api.example.com/search
```

## 📚 Examples

See `examples/cli_usage_examples.sh` for comprehensive usage examples.

Run with:
```bash
./examples/cli_usage_examples.sh
```

## 🆘 Support

### Getting Help

1. **Documentation**: See `CLI_DOCUMENTATION.md` for complete reference
2. **Health Check**: Run `--health-check` to diagnose issues
3. **Examples**: Check `examples/` directory for usage patterns
4. **Configuration**: Review `configs/` for ready-made settings

### Reporting Issues

Include in bug reports:
- Command used
- Error messages
- Health check output
- System information

---

**Author:** Nyimbi Odero (nyimbi@datacraft.co.ke)  
**Company:** Datacraft (www.datacraft.co.ke)  
**Version:** 1.0.0  
**License:** MIT