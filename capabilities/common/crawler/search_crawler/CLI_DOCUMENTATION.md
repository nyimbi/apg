# Search Crawler CLI Documentation

A comprehensive command-line interface for the Search Crawler package with intelligent defaults and deep parameterization capabilities.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Options](#command-line-options)
- [Configuration Files](#configuration-files)
- [Usage Examples](#usage-examples)
- [Modes and Features](#modes-and-features)
- [Output Formats](#output-formats)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

```bash
# Core dependencies
pip install aiohttp beautifulsoup4 lxml requests

# Optional enhanced CLI features
pip install rich click pyyaml

# Optional Crawlee integration
pip install crawlee trafilatura newspaper3k readability-lxml
```

### Setup

```bash
# Make CLI executable
chmod +x search_crawler_cli

# Or run with Python
python cli.py --help
```

## Quick Start

### Basic Search
```bash
# Simple search
./search_crawler_cli "Ethiopia conflict"

# With specific engines
./search_crawler_cli "Somalia security" --engines google,bing,duckduckgo

# Save results to file
./search_crawler_cli "Horn of Africa drought" --output-format json --output-file results.json
```

### Conflict Monitoring
```bash
# Conflict monitoring mode
./search_crawler_cli "Sudan violence" --mode conflict --enable-alerts

# Geographic targeting
./search_crawler_cli "Ethiopia conflict" --mode conflict --target-countries ET,SO,ER
```

### Content Extraction
```bash
# With Crawlee content extraction
./search_crawler_cli "Somalia news" --mode crawlee --extract-content

# High-quality content only
./search_crawler_cli "humanitarian crisis" --mode crawlee --min-quality-score 0.8
```

## Command Line Options

### Core Search Parameters

| Option | Description | Default |
|--------|-------------|---------|
| `query` | Search query (positional argument) | Required |
| `--query`, `-q` | Search query (alternative) | |
| `--mode`, `-m` | Crawler mode: general, conflict, crawlee, auto | auto |
| `--engines`, `-e` | Comma-separated engine list | intelligent selection |
| `--max-results` | Maximum total results | 50 |
| `--max-results-per-engine` | Maximum results per engine | 20 |
| `--list-engines` | List available engines and exit | |

### Content & Extraction

| Option | Description | Default |
|--------|-------------|---------|
| `--download-content` | Download full content from URLs | false |
| `--extract-content` | Extract and analyze content | false |
| `--min-content-length` | Minimum content length for filtering | 100 |
| `--preferred-extraction` | Extraction method: trafilatura, newspaper3k, readability, beautifulsoup | trafilatura |

### Geographic & Conflict Settings

| Option | Description | Default |
|--------|-------------|---------|
| `--target-countries` | Comma-separated ISO country codes | auto for conflict mode |
| `--conflict-regions` | Comma-separated conflict regions | horn_of_africa |
| `--enable-conflict-detection` | Enable conflict keyword detection | true |
| `--escalation-threshold` | Conflict escalation alert threshold (0-1) | 0.7 |

### Performance & Reliability

| Option | Description | Default |
|--------|-------------|---------|
| `--timeout` | Request timeout in seconds | 30 |
| `--max-concurrent` | Maximum concurrent requests | 10 |
| `--retry-attempts` | Number of retry attempts | 3 |
| `--rate-limit-delay` | Delay between requests (seconds) | 1.0 |
| `--enable-stealth` | Enable stealth techniques | true |
| `--disable-stealth` | Disable stealth techniques | |

### Quality & Filtering

| Option | Description | Default |
|--------|-------------|---------|
| `--min-relevance-score` | Minimum relevance score (0-1) | 0.5 |
| `--enable-quality-filtering` | Enable quality-based filtering | true |
| `--min-quality-score` | Minimum quality score for content (0-1) | 0.6 |
| `--deduplicate-results` | Remove duplicate results | true |
| `--no-deduplication` | Disable result deduplication | |

### Output & Reporting

| Option | Description | Default |
|--------|-------------|---------|
| `--output-format`, `-f` | Output format: table, json, csv, yaml, text | table |
| `--output-file`, `-o` | Output file path | stdout |
| `--save-raw` | Save raw results to JSON file | |
| `--verbose`, `-v` | Verbose output | false |
| `--quiet` | Suppress non-essential output | false |
| `--show-progress` | Show progress bar | true |

### Advanced Features

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-alerts` | Enable high-priority alerts | false |
| `--save-config` | Save current configuration to file | |
| `--load-config` | Load configuration from file | |
| `--batch-mode` | Process multiple queries from config | false |
| `--monitor-mode` | Continuous monitoring mode | false |
| `--monitor-interval` | Monitoring interval in seconds | 300 |

### Utility Commands

| Option | Description |
|--------|-------------|
| `--health-check` | Check system health and component availability |
| `--benchmark` | Run performance benchmark |
| `--interactive` | Launch interactive configuration mode |
| `--version` | Show version information |

## Configuration Files

### YAML Configuration Example

```yaml
# search_config.yaml
mode: "conflict"
query: "Ethiopia conflict"
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

### Loading Configuration

```bash
# Load from YAML
./search_crawler_cli --load-config search_config.yaml

# Load and override specific options
./search_crawler_cli --load-config search_config.yaml --max-results 100
```

### Saving Configuration

```bash
# Save current settings
./search_crawler_cli "test query" --save-config my_config.yaml
```

## Usage Examples

### 1. Basic Web Search

```bash
# Simple search with intelligent defaults
./search_crawler_cli "machine learning research 2025"

# Multiple engines with specific format
./search_crawler_cli "AI ethics" --engines google,bing,duckduckgo --output-format csv
```

### 2. Conflict Monitoring

```bash
# Horn of Africa conflict monitoring
./search_crawler_cli "Ethiopia violence" --mode conflict --enable-alerts

# Multi-country conflict analysis
./search_crawler_cli "refugee crisis" --mode conflict \
  --target-countries ET,SO,SS,SD --escalation-threshold 0.8

# Save conflict analysis to file
./search_crawler_cli "Somalia Al-Shabaab" --mode conflict \
  --output-format json --output-file conflict_report.json
```

### 3. Content Extraction

```bash
# High-quality content extraction
./search_crawler_cli "humanitarian aid Horn of Africa" --mode crawlee \
  --extract-content --min-quality-score 0.8

# Specific extraction method
./search_crawler_cli "Somalia news" --mode crawlee \
  --preferred-extraction newspaper3k --download-content
```

### 4. Performance Optimization

```bash
# Fast search with minimal resources
./search_crawler_cli "quick search" --max-results 10 \
  --engines duckduckgo --timeout 15

# Comprehensive search with high concurrency
./search_crawler_cli "detailed analysis" --max-results 200 \
  --max-concurrent 15 --retry-attempts 5
```

### 5. Batch Processing

```bash
# Load batch configuration
./search_crawler_cli --load-config batch_config.yaml --batch-mode

# Monitor mode with alerts
./search_crawler_cli "Sudan conflict" --monitor-mode \
  --monitor-interval 600 --enable-alerts
```

## Modes and Features

### Auto Mode (Default)
- Intelligent engine selection based on query content
- Automatic parameter optimization
- Best general-purpose settings

### General Mode
- Balanced performance for web search
- Broad engine selection
- Standard quality filtering

### Conflict Mode
- Optimized for conflict and security monitoring
- Horn of Africa geographic focus
- Enhanced keyword detection
- Alert system for high-priority events
- Comprehensive engine coverage

### Crawlee Mode
- Advanced content extraction
- Multiple extraction methods with fallbacks
- Quality scoring and filtering
- Full-text analysis capabilities

## Output Formats

### Table (Default)
- Rich formatted table with color coding
- Optimized for terminal viewing
- Shows key metrics and scores

### JSON
- Structured data with full metadata
- Includes search statistics and timestamps
- Ideal for programmatic processing

### CSV
- Spreadsheet-compatible format
- Includes core fields and scores
- Good for data analysis tools

### YAML
- Human-readable structured format
- Maintains data hierarchy
- Easy configuration integration

### Text
- Plain text output
- Compatible with all terminals
- Good for logging and simple processing

## Advanced Features

### Interactive Mode

```bash
# Launch interactive configuration
./search_crawler_cli --interactive
```

Features:
- Step-by-step configuration wizard
- Real-time validation
- Configuration saving
- Immediate search execution

### Health Check

```bash
# Check system status
./search_crawler_cli --health-check
```

Reports:
- Component availability
- Search engine status
- Library dependencies
- Performance capabilities

### Benchmarking

```bash
# Run performance benchmark
./search_crawler_cli --benchmark
```

Tests:
- Search engine performance
- Content extraction speed
- Memory usage
- Concurrent operation efficiency

### Monitoring Mode

```bash
# Continuous monitoring
./search_crawler_cli "Ethiopia crisis" --monitor-mode \
  --monitor-interval 300 --enable-alerts
```

Features:
- Periodic search execution
- Change detection
- Alert generation
- Trend analysis

## Intelligent Defaults

### Engine Selection

| Mode | Default Engines | Rationale |
|------|----------------|-----------|
| General | google, bing, duckduckgo, brave | Balanced coverage and reliability |
| Conflict | google, bing, duckduckgo, yandex, brave, startpage | Comprehensive international perspective |
| Crawlee | google, bing, duckduckgo | Optimized for content extraction |
| Auto | Dynamic based on query | Smart selection based on content analysis |

### Performance Tuning

| Parameter | General | Conflict | Crawlee | Rationale |
|-----------|---------|----------|---------|-----------|
| Timeout | 30s | 45s | 60s | Increased for complex operations |
| Concurrency | 10 | 8 | 5 | Balanced with reliability needs |
| Rate Limit | 1.0s | 2.0s | 2.0s | Respectful of service limits |
| Retries | 3 | 5 | 3 | Higher for critical monitoring |

### Quality Thresholds

| Setting | General | Conflict | Crawlee | Purpose |
|---------|---------|----------|---------|---------|
| Min Relevance | 0.5 | 0.6 | 0.7 | Increasing strictness |
| Min Quality | 0.6 | 0.7 | 0.8 | Higher standards for analysis |
| Content Length | 100 | 200 | 500 | More substantial content |

## Error Handling

### Common Issues

1. **No results found**
   - Try broader search terms
   - Reduce quality thresholds
   - Check engine availability

2. **Timeout errors**
   - Increase timeout setting
   - Reduce concurrency
   - Use fewer engines

3. **Rate limiting**
   - Increase rate limit delay
   - Reduce concurrent requests
   - Enable stealth mode

4. **Import errors**
   - Check dependencies installation
   - Verify Python path
   - Run health check

### Debug Mode

```bash
# Enable verbose logging
./search_crawler_cli "test" --verbose

# Check component status
./search_crawler_cli --health-check
```

## Configuration Examples

### Complete Files

See the `configs/` directory for comprehensive examples:
- `example_general_search.yaml` - Basic web search
- `example_conflict_monitoring.yaml` - Security monitoring
- `example_crawlee_enhanced.yaml` - Content extraction
- `example_monitoring_batch.yaml` - Batch processing

### Environment Variables

```bash
# Set default configuration
export SEARCH_CRAWLER_CONFIG="./default_config.yaml"
export SEARCH_CRAWLER_ENGINES="google,bing,duckduckgo"
export SEARCH_CRAWLER_MODE="conflict"
```

## Integration

### Shell Scripts

```bash
#!/bin/bash
# monitor_conflicts.sh
./search_crawler_cli "$1" --mode conflict --enable-alerts \
  --output-format json --output-file "alert_$(date +%Y%m%d_%H%M%S).json"
```

### Python Integration

```python
import subprocess
import json

# Run CLI and capture results
result = subprocess.run([
    './search_crawler_cli', 'Ethiopia conflict',
    '--mode', 'conflict',
    '--output-format', 'json'
], capture_output=True, text=True)

if result.returncode == 0:
    data = json.loads(result.stdout)
    print(f"Found {len(data['results'])} results")
```

### API Integration

```bash
# Generate JSON for API consumption
./search_crawler_cli "API query" --output-format json | \
  curl -X POST -H "Content-Type: application/json" \
  -d @- https://api.example.com/search-results
```

## Performance Tips

1. **Engine Selection**
   - Use 3-5 engines for optimal speed/coverage balance
   - Prefer reliable engines (google, bing, duckduckgo)

2. **Concurrency**
   - Start with default concurrency (10)
   - Reduce if experiencing timeouts
   - Increase cautiously for faster hardware

3. **Quality Settings**
   - Lower thresholds for broader results
   - Higher thresholds for precision
   - Enable filtering only when needed

4. **Content Extraction**
   - Use only when necessary (slower)
   - Prefer trafilatura for speed and quality
   - Set appropriate minimum content length

## Security Considerations

1. **Stealth Mode**
   - Enabled by default for respectful crawling
   - Use random user agents and delays
   - Respect robots.txt when possible

2. **Rate Limiting**
   - Default delays prevent service overload
   - Adjust based on service tolerance
   - Monitor for 429 (rate limit) responses

3. **Data Privacy**
   - Results contain public web data only
   - No personal information collection
   - Respect website terms of service

## Support and Contributing

### Getting Help

1. Check this documentation
2. Run `--health-check` for system status
3. Use `--verbose` for detailed logging
4. Review configuration examples

### Reporting Issues

Include in bug reports:
- CLI command used
- Error messages
- System information (`--health-check` output)
- Configuration file (if used)

### Contributing

1. Test new features with `--benchmark`
2. Update documentation for new options
3. Follow existing code style
4. Add configuration examples for new features

---

**Author:** Nyimbi Odero (nyimbi@datacraft.co.ke)  
**Company:** Datacraft (www.datacraft.co.ke)  
**Version:** 1.0.0  
**License:** MIT