# Gen Crawler Architecture

Comprehensive architectural documentation for the gen_crawler package, detailing design patterns, component interactions, and extensibility mechanisms.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Commands  │ │  Exporters  │ │      Utils          │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────┐
│                  Core Engine                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │ GenCrawler  │ │  Adaptive   │ │   Site Profiling    │ │
│  │             │ │  Strategy   │ │                     │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────┐
│               Crawlee Integration                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Adaptive  │ │   Browser   │ │    HTTP Client      │ │
│  │ Playwright  │ │ Automation  │ │                     │ │
│  │  Crawler    │ │             │ │                     │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────┐
│              Content Processing                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │   Multi-    │ │   Content   │ │     Quality         │ │
│  │  Method     │ │ Analysis &  │ │    Scoring          │ │
│  │ Extraction  │ │ Classification │ │                   │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────┐
│              Configuration & Storage                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │Type-Safe    │ │ Database    │ │   Export System     │ │
│  │Configuration│ │Integration  │ │                     │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🧩 Core Components

### 1. GenCrawler (Main Engine)

**Location**: `core/gen_crawler.py`

The central orchestrator that coordinates all crawling activities.

```python
class GenCrawler:
    """
    Main crawler engine using Crawlee's AdaptivePlaywrightCrawler.
    
    Key Responsibilities:
    - Initialize Crawlee AdaptivePlaywrightCrawler
    - Manage crawl sessions and state
    - Coordinate with adaptive strategy manager
    - Handle content extraction and analysis
    - Provide statistics and monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crawler = None  # AdaptivePlaywrightCrawler instance
        self.current_site_result = None
        self.visited_urls = set()
        self.stats = {...}
```

**Key Methods**:
- `initialize()`: Set up Crawlee crawler with router
- `crawl_site(url)`: Execute full-site crawling
- `_handle_request(context)`: Process individual pages
- `cleanup()`: Resource cleanup and state reset

### 2. AdaptiveCrawler (Strategy Management)

**Location**: `core/adaptive_crawler.py`

Intelligent strategy management for optimizing crawling behavior.

```python
class AdaptiveCrawler:
    """
    Adaptive strategy manager that optimizes crawling behavior
    based on site characteristics and performance data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.site_profiles: Dict[str, SiteProfile] = {}
        self.global_stats = {...}
    
    def recommend_strategy(self, url: str) -> CrawlStrategy:
        """Recommend optimal strategy based on site profile."""
        
    def update_strategy_performance(self, url: str, strategy: CrawlStrategy, 
                                  success_rate: float, load_time: float):
        """Update performance metrics for continuous optimization."""
```

**Strategy Types**:
- `ADAPTIVE`: Let Crawlee decide (default)
- `HTTP_ONLY`: Force HTTP-only crawling
- `BROWSER_ONLY`: Force browser-based crawling
- `MIXED`: Alternate between strategies

### 3. Site Profiling System

**Location**: `core/adaptive_crawler.py`

Maintains detailed profiles of crawled sites for optimization.

```python
@dataclass
class SiteProfile:
    """Profile information for a crawled site."""
    domain: str
    total_pages: int = 0
    successful_pages: int = 0
    average_load_time: float = 0.0
    preferred_strategy: CrawlStrategy = CrawlStrategy.ADAPTIVE
    requires_javascript: bool = False
    has_infinite_scroll: bool = False
    cloudflare_protection: bool = False
    rate_limit_detected: bool = False
    performance_score: float = 0.0
    crawl_history: List[Dict[str, Any]] = field(default_factory=list)
```

**Profile Features**:
- Performance tracking and scoring
- Site characteristic detection
- Strategy recommendation based on history
- Automatic optimization thresholds

## 🔧 Content Processing Pipeline

### 1. Multi-Method Content Extraction

**Location**: `parsers/content_parser.py`

```python
class GenContentParser:
    """
    Advanced content parser using multiple extraction methods.
    """
    
    def parse_content(self, url: str, html_content: str) -> ParsedSiteContent:
        """
        Parse content using best available method:
        1. Trafilatura (preferred for news/articles)
        2. Newspaper3k (article-specific)
        3. Readability (clean content extraction)
        4. BeautifulSoup (fallback)
        5. Basic regex (ultimate fallback)
        """
```

**Extraction Method Priority**:
1. **Trafilatura**: Best for news and article content
2. **Newspaper3k**: Specialized for news articles
3. **Readability**: Good for general content cleaning
4. **BeautifulSoup**: Reliable fallback parser
5. **Basic Parsing**: Regex-based ultimate fallback

### 2. Content Analysis & Classification

```python
class ContentAnalyzer:
    """Analyzes and scores content quality and relevance."""
    
    def analyze_content_type(self, url: str, title: str, content: str) -> str:
        """
        Classify content type:
        - article: News articles, blog posts, stories
        - page: Static pages, about pages
        - listing: Category pages, archives
        - snippet: Short content, excerpts
        - insufficient_content: Too little content
        """
    
    def calculate_quality_score(self, parsed_content: ParsedSiteContent) -> float:
        """
        Calculate quality score (0.0-1.0) based on:
        - Title quality (length, descriptiveness)
        - Content length and structure
        - Metadata presence (authors, dates, keywords)
        - Language detection and readability
        - HTML structure indicators
        """
```

### 3. Quality Scoring Algorithm

```python
def calculate_quality_score(self, parsed_content, html_content=""):
    score = 0.0
    
    # Title quality (0.2 max)
    title_score = min(len(title) / 60, 1.0) * 0.2
    
    # Content length quality (0.3 max)
    if word_count >= 300:
        length_score = min(word_count / 2000, 1.0) * 0.3
    
    # Structure quality (0.2 max)
    paragraphs = content.count('\n\n')
    sentences = content.count('. ')
    structure_score = min((paragraphs + sentences) / 20, 1.0) * 0.2
    
    # Metadata quality (0.2 max)
    metadata_score = 0.05 * (has_authors + has_date + has_keywords + has_metadata)
    
    # Language and readability (0.1 max)
    readability_score = 0.1 if language != "unknown" else 0.05
    
    return max(0.0, min(score, 1.0))
```

## ⚙️ Configuration Architecture

### 1. Type-Safe Configuration System

**Location**: `config/gen_config.py`

```python
@dataclass
class GenCrawlerSettings:
    """Complete settings for GenCrawler."""
    content_filters: ContentFilterConfig = field(default_factory=ContentFilterConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    stealth: StealthConfig = field(default_factory=StealthConfig)
```

### 2. Configuration Hierarchy

```
Configuration Priority (highest to lowest):
1. CLI Arguments          # --max-pages 100
2. Environment Variables  # GEN_CRAWLER_MAX_PAGES=100
3. Configuration File     # config.json
4. Programmatic Settings  # config.settings.performance.max_pages = 100
5. Default Values         # Built-in defaults
```

### 3. Configuration Templates

```python
def create_news_config():
    """News-optimized configuration."""
    config = create_gen_config()
    config.settings.content_filters.include_patterns = [
        'article', 'news', 'story', 'breaking'
    ]
    config.settings.performance.crawl_delay = 3.0
    return config

def create_research_config():
    """Research-optimized configuration."""
    config = create_gen_config()
    config.settings.performance.max_pages_per_site = 1000
    config.settings.content_filters.min_content_length = 500
    return config
```

## 🎨 CLI Architecture

### 1. Command Structure

```python
# CLI Command Hierarchy
gen-crawler
├── crawl           # Primary crawling command
│   ├── URLs        # Target URLs (required)
│   ├── --output    # Output directory
│   ├── --format    # Export format
│   └── [50+ options]
├── analyze         # Analyze existing crawl data
├── config          # Configuration management
│   ├── --create    # Create new config
│   ├── --validate  # Validate existing config
│   └── --template  # Use template (news, research, etc.)
└── export          # Export data to different formats
    ├── --format    # Target format
    ├── --filter-*  # Content filtering
    └── --organize-by # Organization method
```

### 2. Export System Architecture

```python
class BaseExporter:
    """Base class for all exporters."""
    
class MarkdownExporter(BaseExporter):
    """
    Export as clean markdown files:
    - Organized directory structure
    - Clean content formatting
    - Metadata preservation
    - Index file generation
    """
    
class JSONExporter(BaseExporter):
    """Export as structured JSON with optional compression."""
    
class CSVExporter(BaseExporter):
    """Export as tabular data for analysis."""
    
class HTMLExporter(BaseExporter):
    """Export as styled HTML with navigation."""
```

## 🔄 Data Flow Architecture

### 1. Crawling Data Flow

```
URL Input → URL Validation → Crawlee AdaptivePlaywrightCrawler
    ↓
Site Profiling ← Strategy Selection ← Site Characteristics Detection
    ↓
Page Processing → Content Extraction → Quality Analysis
    ↓
Content Classification → Metadata Extraction → Link Discovery
    ↓
Result Aggregation → Statistics Update → Export Processing
    ↓
Storage (Files/Database) ← Export Formatting ← Result Serialization
```

### 2. Adaptive Strategy Flow

```
New Site → Profile Creation → Default Strategy (ADAPTIVE)
    ↓
Crawl Execution → Performance Monitoring → Characteristic Detection
    ↓
Profile Update ← Performance Analysis ← Success Rate Calculation
    ↓
Strategy Optimization → Performance Threshold Check → Strategy Update
    ↓
Next Crawl → Updated Strategy → Improved Performance
```

### 3. Content Processing Flow

```
Raw HTML → Parser Selection → Method Prioritization
    ↓
Content Extraction → Cleaning & Formatting → Structure Analysis
    ↓
Quality Scoring → Content Classification → Metadata Extraction
    ↓
Link Analysis → Image Extraction → Keyword Extraction
    ↓
Final Result → Serialization → Storage/Export
```

## 🏛️ Design Patterns

### 1. Strategy Pattern (Adaptive Crawling)

```python
class CrawlStrategy(Enum):
    ADAPTIVE = "adaptive"
    HTTP_ONLY = "http_only"
    BROWSER_ONLY = "browser_only"
    MIXED = "mixed"

class AdaptiveCrawler:
    def recommend_strategy(self, url: str) -> CrawlStrategy:
        """Strategy selection based on site characteristics."""
        profile = self.get_site_profile(url)
        return self._analyze_site_characteristics(profile)
```

### 2. Factory Pattern (Configuration & Components)

```python
def create_gen_crawler(config: Optional[Dict[str, Any]] = None) -> GenCrawler:
    """Factory function for GenCrawler creation."""
    
def create_gen_config(config_file: Optional[Path] = None) -> GenCrawlerConfig:
    """Factory function for configuration creation."""
    
def create_content_parser(config: Optional[Dict[str, Any]] = None) -> GenContentParser:
    """Factory function for content parser creation."""
```

### 3. Observer Pattern (Progress Monitoring)

```python
class CrawlProgressObserver:
    def on_page_crawled(self, page_result: GenCrawlResult):
        """Called when a page is successfully crawled."""
        
    def on_site_completed(self, site_result: GenSiteResult):
        """Called when site crawling is completed."""
        
    def on_error(self, error: Exception, context: Dict[str, Any]):
        """Called when an error occurs."""
```

### 4. Command Pattern (CLI Commands)

```python
class BaseCommand:
    async def execute(self, args: argparse.Namespace) -> None:
        """Execute the command."""
        
class CrawlCommand(BaseCommand):
    async def execute(self, args: argparse.Namespace) -> None:
        """Execute crawl command."""
        
class ExportCommand(BaseCommand):
    async def execute(self, args: argparse.Namespace) -> None:
        """Execute export command."""
```

## 🔌 Extensibility Mechanisms

### 1. Custom Content Extractors

```python
class CustomExtractor:
    def extract_content(self, url: str, html_content: str) -> ParsedSiteContent:
        """Custom extraction logic."""
        
# Register custom extractor
parser = GenContentParser()
parser.register_extractor('custom', CustomExtractor())
```

### 2. Custom Export Formats

```python
class CustomExporter(BaseExporter):
    async def export_results(self, results: List[Any]) -> None:
        """Custom export implementation."""
        
# Use custom exporter
exporter = CustomExporter(output_dir="./custom_output")
await exporter.export_results(crawl_results)
```

### 3. Custom Analysis Plugins

```python
class ConflictAnalysisPlugin:
    def analyze_content(self, content: ParsedSiteContent) -> Dict[str, Any]:
        """Custom conflict analysis."""
        
# Register plugin
analyzer = ContentAnalyzer()
analyzer.register_plugin('conflict', ConflictAnalysisPlugin())
```

## 📊 Performance Architecture

### 1. Async/Await Architecture

```python
# All I/O operations are async
async def crawl_site(self, base_url: str) -> GenSiteResult:
    await self.initialize()
    await self.crawler.run([base_url])
    await self.cleanup()

# Concurrent processing
async def crawl_multiple_sites(urls: List[str]) -> List[GenSiteResult]:
    tasks = [self.crawl_site(url) for url in urls]
    return await asyncio.gather(*tasks)
```

### 2. Resource Management

```python
class ResourceManager:
    def __init__(self, memory_limit_mb: int = 1024):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.current_usage = 0
    
    async def __aenter__(self):
        """Async context manager entry."""
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources on exit."""
```

### 3. Caching Strategy

```python
class AdaptiveCache:
    """
    Multi-level caching:
    1. In-memory cache for site profiles
    2. Disk cache for parsed content
    3. HTTP cache for network requests
    """
    
    def get_site_profile(self, domain: str) -> Optional[SiteProfile]:
        """Get cached site profile."""
        
    def cache_parsed_content(self, url: str, content: ParsedSiteContent):
        """Cache parsed content for reuse."""
```

## 🔒 Security Architecture

### 1. Input Validation

```python
def validate_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Comprehensive URL validation:
    - Protocol validation (http/https only)
    - Domain validation
    - Path sanitization
    - Query parameter filtering
    """

def sanitize_filename(filename: str) -> str:
    """
    Filename sanitization:
    - Remove invalid characters
    - Prevent directory traversal
    - Length limiting
    """
```

### 2. Rate Limiting & Respect

```python
class RespectfulCrawler:
    """
    Implements crawling best practices:
    - robots.txt compliance
    - Rate limiting per domain
    - User-agent identification
    - Retry backoff
    """
    
    def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        
    def apply_rate_limit(self, domain: str):
        """Apply appropriate rate limiting."""
```

### 3. Content Sanitization

```python
def sanitize_content(content: str) -> str:
    """
    Content sanitization:
    - HTML entity decoding
    - Script/style removal
    - Malicious content filtering
    - Encoding normalization
    """
```

This architecture provides a robust, scalable, and maintainable foundation for the gen_crawler package while enabling easy extension and customization for specific use cases.