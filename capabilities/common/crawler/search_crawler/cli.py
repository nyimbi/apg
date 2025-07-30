#!/usr/bin/env python3
"""
Search Crawler CLI
==================

Comprehensive command-line interface for the Search Crawler package with intelligent 
defaults and deep parameterization. Supports all crawler types, search engines, 
and configuration options.

Features:
- Multi-mode operation (general, conflict, crawlee-enhanced)
- Intelligent defaults based on use case
- Deep configuration of all parameters
- Output formatting (JSON, CSV, text)
- Batch processing and configuration files
- Real-time monitoring and progress tracking
- Export capabilities

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import argparse
import asyncio
import json
import csv
import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import yaml
from dataclasses import dataclass, asdict
from enum import Enum

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# CLI-specific imports
try:
    import click
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Confirm, Prompt, IntPrompt
    CLI_ENHANCED = True
except ImportError:
    CLI_ENHANCED = False
    Console = None

# Search crawler imports
try:
    # Try absolute imports first
    try:
        from search_crawler.core.search_crawler import SearchCrawler, SearchCrawlerConfig
        from search_crawler.core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
        from search_crawler.core.crawlee_enhanced_search_crawler import (
            CrawleeEnhancedSearchCrawler, CrawleeSearchConfig, 
            create_crawlee_search_config, CRAWLEE_AVAILABLE
        )
        from search_crawler.engines import SEARCH_ENGINES, get_available_engines
        from search_crawler.keywords.conflict_keywords import ConflictKeywordManager
        from search_crawler.keywords.horn_of_africa_keywords import HornOfAfricaKeywords
    except ImportError:
        # Fallback to relative imports
        from core.search_crawler import SearchCrawler, SearchCrawlerConfig
        from core.conflict_search_crawler import ConflictSearchCrawler, ConflictSearchConfig
        from core.crawlee_enhanced_search_crawler import (
            CrawleeEnhancedSearchCrawler, CrawleeSearchConfig, 
            create_crawlee_search_config, CRAWLEE_AVAILABLE
        )
        from engines import SEARCH_ENGINES, get_available_engines
        from keywords.conflict_keywords import ConflictKeywordManager
        from keywords.horn_of_africa_keywords import HornOfAfricaKeywords
    
    CRAWLER_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"Search crawler components not available: {e}")
    CRAWLER_AVAILABLE = False
    # Create dummy classes to prevent further errors
    class SearchCrawler: pass
    class SearchCrawlerConfig: pass
    class ConflictSearchCrawler: pass
    class ConflictSearchConfig: pass
    class CrawleeEnhancedSearchCrawler: pass
    class CrawleeSearchConfig: pass
    SEARCH_ENGINES = {}
    CRAWLEE_AVAILABLE = False
    def get_available_engines(): return ["duckduckgo"]
    def create_crawlee_search_config(**kwargs): return None


class CrawlerMode(Enum):
    """Available crawler modes."""
    GENERAL = "general"
    CONFLICT = "conflict"
    CRAWLEE = "crawlee"
    AUTO = "auto"


class OutputFormat(Enum):
    """Available output formats."""
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    YAML = "yaml"
    TABLE = "table"


@dataclass
class CLIConfig:
    """CLI configuration with intelligent defaults."""
    
    # Core search parameters
    mode: str = "auto"
    query: str = ""
    engines: List[str] = None
    max_results: int = 50
    max_results_per_engine: int = 20
    
    # Content and extraction
    download_content: bool = False
    extract_content: bool = False
    min_content_length: int = 100
    preferred_extraction_method: str = "trafilatura"
    
    # Geographic and conflict settings
    target_countries: List[str] = None
    conflict_regions: List[str] = None
    enable_conflict_detection: bool = True
    escalation_threshold: float = 0.7
    
    # Performance and reliability
    timeout: int = 30
    max_concurrent: int = 10
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    enable_stealth: bool = True
    
    # Quality and filtering
    min_relevance_score: float = 0.5
    enable_quality_filtering: bool = True
    min_quality_score: float = 0.6
    deduplicate_results: bool = True
    
    # Output and reporting
    output_format: str = "table"
    output_file: str = ""
    verbose: bool = False
    quiet: bool = False
    
    # Advanced features
    enable_alerts: bool = False
    save_config: str = ""
    load_config: str = ""
    batch_mode: bool = False
    monitor_mode: bool = False
    
    def __post_init__(self):
        """Apply intelligent defaults based on configuration."""
        if self.engines is None:
            self.engines = self._get_default_engines()
        
        if self.target_countries is None and self.mode in ["conflict", "auto"]:
            self.target_countries = ["ET", "SO", "ER", "DJ", "SD", "SS", "KE", "UG"]
        
        if self.conflict_regions is None and self.mode in ["conflict", "auto"]:
            self.conflict_regions = ["horn_of_africa"]
    
    def _get_default_engines(self) -> List[str]:
        """Get intelligent engine defaults based on mode."""
        if not CRAWLER_AVAILABLE:
            return ["duckduckgo"]
        
        available = get_available_engines()
        
        if self.mode == "general":
            # Balanced set for general search
            preferred = ["google", "bing", "duckduckgo", "brave"]
        elif self.mode == "conflict":
            # Comprehensive set for conflict monitoring
            preferred = ["google", "bing", "duckduckgo", "yandex", "brave", "startpage"]
        elif self.mode == "crawlee":
            # Optimized for content extraction
            preferred = ["google", "bing", "duckduckgo"]
        else:  # auto mode
            # Smart default based on query content
            if any(word in self.query.lower() for word in ["conflict", "violence", "war", "crisis"]):
                preferred = ["google", "bing", "duckduckgo", "yandex", "brave"]
            else:
                preferred = ["google", "bing", "duckduckgo"]
        
        return [engine for engine in preferred if engine in available][:6]  # Max 6 engines


class SearchCrawlerCLI:
    """Comprehensive CLI for Search Crawler package."""
    
    def __init__(self):
        self.console = Console() if CLI_ENHANCED else None
        self.config = CLIConfig()
        self.results = []
        self.crawler = None
        self.start_time = None
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        if self.config.quiet:
            log_level = logging.WARNING
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr) if not self.config.quiet else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def print_message(self, message: str, style: str = ""):
        """Print message with optional styling."""
        if self.config.quiet:
            return
        
        if self.console and style:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"❌ Error: {message}", style="bold red")
        else:
            print(f"Error: {message}", file=sys.stderr)
    
    def print_warning(self, message: str):
        """Print warning message."""
        if not self.config.quiet:
            if self.console:
                self.console.print(f"⚠️  Warning: {message}", style="bold yellow")
            else:
                print(f"Warning: {message}")
    
    def print_success(self, message: str):
        """Print success message."""
        if self.console:
            self.console.print(f"✅ {message}", style="bold green")
        else:
            print(f"Success: {message}")
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser."""
        parser = argparse.ArgumentParser(
            description="Search Crawler CLI - Multi-engine search with content analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic search
  %(prog)s "Ethiopia conflict" --mode conflict --max-results 20

  # Advanced conflict monitoring
  %(prog)s "Horn of Africa violence" --mode conflict --engines google,bing,yandex \\
    --extract-content --target-countries ET,SO,ER --enable-alerts

  # Content extraction with Crawlee
  %(prog)s "Somalia security" --mode crawlee --download-content \\
    --preferred-extraction trafilatura --min-quality-score 0.8

  # Batch processing from config
  %(prog)s --load-config ./search_config.yaml --batch-mode

  # Real-time monitoring
  %(prog)s "Sudan conflict" --monitor-mode --enable-alerts \\
    --output-format json --output-file alerts.json

For more information: https://github.com/your-repo/search-crawler
            """
        )
        
        # Core search parameters
        search_group = parser.add_argument_group('Search Parameters')
        search_group.add_argument(
            'query', 
            nargs='?',
            help='Search query (can also be provided via --query)'
        )
        search_group.add_argument(
            '-q', '--query',
            help='Search query'
        )
        search_group.add_argument(
            '-m', '--mode',
            choices=[mode.value for mode in CrawlerMode],
            default='auto',
            help='Crawler mode (default: auto - intelligent detection)'
        )
        search_group.add_argument(
            '-e', '--engines',
            help='Comma-separated list of search engines (default: intelligent selection)'
        )
        search_group.add_argument(
            '--list-engines',
            action='store_true',
            help='List available search engines and exit'
        )
        search_group.add_argument(
            '--max-results',
            type=int,
            default=50,
            help='Maximum total results (default: 50)'
        )
        search_group.add_argument(
            '--max-results-per-engine',
            type=int,
            default=20,
            help='Maximum results per engine (default: 20)'
        )
        
        # Content and extraction
        content_group = parser.add_argument_group('Content & Extraction')
        content_group.add_argument(
            '--download-content',
            action='store_true',
            help='Download full content from result URLs'
        )
        content_group.add_argument(
            '--extract-content',
            action='store_true',
            help='Extract and analyze content using multiple methods'
        )
        content_group.add_argument(
            '--min-content-length',
            type=int,
            default=100,
            help='Minimum content length for filtering (default: 100)'
        )
        content_group.add_argument(
            '--preferred-extraction',
            choices=['trafilatura', 'newspaper3k', 'readability', 'beautifulsoup'],
            default='trafilatura',
            help='Preferred content extraction method (default: trafilatura)'
        )
        
        # Geographic and conflict settings
        geo_group = parser.add_argument_group('Geographic & Conflict Settings')
        geo_group.add_argument(
            '--target-countries',
            help='Comma-separated ISO country codes (e.g., ET,SO,ER)'
        )
        geo_group.add_argument(
            '--conflict-regions',
            help='Comma-separated conflict regions (default: horn_of_africa)'
        )
        geo_group.add_argument(
            '--enable-conflict-detection',
            action='store_true',
            default=True,
            help='Enable conflict keyword detection (default: true)'
        )
        geo_group.add_argument(
            '--escalation-threshold',
            type=float,
            default=0.7,
            help='Conflict escalation alert threshold (default: 0.7)'
        )
        
        # Performance and reliability
        perf_group = parser.add_argument_group('Performance & Reliability')
        perf_group.add_argument(
            '--timeout',
            type=int,
            default=30,
            help='Request timeout in seconds (default: 30)'
        )
        perf_group.add_argument(
            '--max-concurrent',
            type=int,
            default=10,
            help='Maximum concurrent requests (default: 10)'
        )
        perf_group.add_argument(
            '--retry-attempts',
            type=int,
            default=3,
            help='Number of retry attempts (default: 3)'
        )
        perf_group.add_argument(
            '--rate-limit-delay',
            type=float,
            default=1.0,
            help='Delay between requests in seconds (default: 1.0)'
        )
        perf_group.add_argument(
            '--enable-stealth',
            action='store_true',
            default=True,
            help='Enable stealth techniques (default: true)'
        )
        perf_group.add_argument(
            '--disable-stealth',
            action='store_true',
            help='Disable stealth techniques'
        )
        
        # Quality and filtering
        quality_group = parser.add_argument_group('Quality & Filtering')
        quality_group.add_argument(
            '--min-relevance-score',
            type=float,
            default=0.5,
            help='Minimum relevance score (default: 0.5)'
        )
        quality_group.add_argument(
            '--enable-quality-filtering',
            action='store_true',
            default=True,
            help='Enable quality-based filtering (default: true)'
        )
        quality_group.add_argument(
            '--min-quality-score',
            type=float,
            default=0.6,
            help='Minimum quality score for content (default: 0.6)'
        )
        quality_group.add_argument(
            '--deduplicate-results',
            action='store_true',
            default=True,
            help='Remove duplicate results (default: true)'
        )
        quality_group.add_argument(
            '--no-deduplication',
            action='store_true',
            help='Disable result deduplication'
        )
        
        # Output and reporting
        output_group = parser.add_argument_group('Output & Reporting')
        output_group.add_argument(
            '-f', '--output-format',
            choices=[fmt.value for fmt in OutputFormat],
            default='table',
            help='Output format (default: table)'
        )
        output_group.add_argument(
            '-o', '--output-file',
            help='Output file path'
        )
        output_group.add_argument(
            '--save-raw',
            help='Save raw results to file (JSON format)'
        )
        output_group.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Verbose output'
        )
        output_group.add_argument(
            '--quiet',
            action='store_true',
            help='Suppress non-essential output'
        )
        output_group.add_argument(
            '--show-progress',
            action='store_true',
            default=True,
            help='Show progress bar (default: true)'
        )
        
        # Advanced features
        advanced_group = parser.add_argument_group('Advanced Features')
        advanced_group.add_argument(
            '--enable-alerts',
            action='store_true',
            help='Enable high-priority alerts'
        )
        advanced_group.add_argument(
            '--save-config',
            help='Save current configuration to file'
        )
        advanced_group.add_argument(
            '--load-config',
            help='Load configuration from file (YAML/JSON)'
        )
        advanced_group.add_argument(
            '--batch-mode',
            action='store_true',
            help='Process multiple queries from config file'
        )
        advanced_group.add_argument(
            '--monitor-mode',
            action='store_true',
            help='Continuous monitoring mode'
        )
        advanced_group.add_argument(
            '--monitor-interval',
            type=int,
            default=300,
            help='Monitoring interval in seconds (default: 300)'
        )
        
        # Utility commands
        utility_group = parser.add_argument_group('Utility Commands')
        utility_group.add_argument(
            '--health-check',
            action='store_true',
            help='Check system health and component availability'
        )
        utility_group.add_argument(
            '--benchmark',
            action='store_true',
            help='Run performance benchmark'
        )
        utility_group.add_argument(
            '--interactive',
            action='store_true',
            help='Launch interactive configuration mode'
        )
        utility_group.add_argument(
            '--version',
            action='version',
            version='Search Crawler CLI v1.0.0'
        )
        
        return parser
    
    def load_configuration(self, config_file: str):
        """Load configuration from file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.print_error(f"Configuration file not found: {config_file}")
                return False
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    self.print_error(f"Unsupported config format: {config_path.suffix}")
                    return False
            
            # Update configuration
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.print_success(f"Configuration loaded from {config_file}")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to load configuration: {e}")
            return False
    
    def save_configuration(self, config_file: str):
        """Save current configuration to file."""
        try:
            config_path = Path(config_file)
            config_data = asdict(self.config)
            
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    self.print_error(f"Unsupported config format: {config_path.suffix}")
                    return False
            
            self.print_success(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to save configuration: {e}")
            return False
    
    def parse_arguments(self, args=None):
        """Parse command line arguments."""
        parser = self.create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Handle utility commands first
        if parsed_args.list_engines:
            self.list_engines()
            sys.exit(0)
        
        if parsed_args.health_check:
            self.health_check()
            sys.exit(0)
        
        if parsed_args.interactive:
            self.interactive_mode()
            sys.exit(0)
        
        # Load configuration if specified
        if parsed_args.load_config:
            if not self.load_configuration(parsed_args.load_config):
                sys.exit(1)
        
        # Update configuration from arguments
        self.update_config_from_args(parsed_args)
        
        # Validate configuration
        if not self.validate_configuration():
            sys.exit(1)
        
        return parsed_args
    
    def update_config_from_args(self, args):
        """Update configuration from parsed arguments."""
        # Core parameters
        if args.query or args.query:
            self.config.query = args.query or args.query
        
        self.config.mode = args.mode
        
        if args.engines:
            self.config.engines = [e.strip() for e in args.engines.split(',')]
        
        self.config.max_results = args.max_results
        self.config.max_results_per_engine = args.max_results_per_engine
        
        # Content and extraction
        self.config.download_content = args.download_content
        self.config.extract_content = args.extract_content
        self.config.min_content_length = args.min_content_length
        self.config.preferred_extraction_method = args.preferred_extraction
        
        # Geographic settings
        if args.target_countries:
            self.config.target_countries = [c.strip().upper() for c in args.target_countries.split(',')]
        
        if args.conflict_regions:
            self.config.conflict_regions = [r.strip() for r in args.conflict_regions.split(',')]
        
        self.config.enable_conflict_detection = args.enable_conflict_detection
        self.config.escalation_threshold = args.escalation_threshold
        
        # Performance
        self.config.timeout = args.timeout
        self.config.max_concurrent = args.max_concurrent
        self.config.retry_attempts = args.retry_attempts
        self.config.rate_limit_delay = args.rate_limit_delay
        
        if args.disable_stealth:
            self.config.enable_stealth = False
        elif args.enable_stealth:
            self.config.enable_stealth = True
        
        # Quality
        self.config.min_relevance_score = args.min_relevance_score
        self.config.enable_quality_filtering = args.enable_quality_filtering
        self.config.min_quality_score = args.min_quality_score
        
        if args.no_deduplication:
            self.config.deduplicate_results = False
        elif args.deduplicate_results:
            self.config.deduplicate_results = True
        
        # Output
        self.config.output_format = args.output_format
        self.config.output_file = args.output_file or ""
        self.config.verbose = args.verbose
        self.config.quiet = args.quiet
        
        # Advanced
        self.config.enable_alerts = args.enable_alerts
        self.config.save_config = args.save_config or ""
        self.config.batch_mode = args.batch_mode
        self.config.monitor_mode = args.monitor_mode
        
        # Re-apply intelligent defaults
        self.config.__post_init__()
    
    def validate_configuration(self) -> bool:
        """Validate configuration parameters."""
        if not self.config.query and not self.config.batch_mode and not self.config.monitor_mode:
            self.print_error("Search query is required (use --query or provide as positional argument)")
            return False
        
        if not CRAWLER_AVAILABLE:
            self.print_error("Search crawler components are not available")
            return False
        
        if self.config.mode == "crawlee" and not CRAWLEE_AVAILABLE:
            self.print_warning("Crawlee mode requested but Crawlee library not available, falling back to general mode")
            self.config.mode = "general"
        
        if self.config.max_results <= 0:
            self.print_error("Max results must be positive")
            return False
        
        if self.config.escalation_threshold < 0 or self.config.escalation_threshold > 1:
            self.print_error("Escalation threshold must be between 0 and 1")
            return False
        
        return True
    
    def list_engines(self):
        """List available search engines."""
        if not CRAWLER_AVAILABLE:
            self.print_error("Search crawler components not available")
            return
        
        available = get_available_engines()
        all_engines = list(SEARCH_ENGINES.keys())
        
        if self.console:
            table = Table(title="Available Search Engines")
            table.add_column("Engine", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Description", style="white")
            
            descriptions = {
                'google': 'Google Search - Comprehensive results',
                'bing': 'Microsoft Bing - Alternative perspective',
                'duckduckgo': 'DuckDuckGo - Privacy-focused',
                'yandex': 'Yandex - Russian/Eastern European focus',
                'baidu': 'Baidu - Chinese search engine',
                'yahoo': 'Yahoo Search - Portal-based results',
                'startpage': 'Startpage - Private Google results',
                'searx': 'SearX - Open source metasearch',
                'brave': 'Brave Search - Independent index',
                'mojeek': 'Mojeek - Independent crawler',
                'swisscows': 'Swisscows - Semantic search'
            }
            
            for engine in all_engines:
                status = "✅ Available" if engine in available else "❌ Unavailable"
                desc = descriptions.get(engine, "Search engine")
                table.add_row(engine.title(), status, desc)
            
            self.console.print(table)
        else:
            print("Available Search Engines:")
            print("-" * 40)
            for engine in all_engines:
                status = "✅" if engine in available else "❌"
                print(f"{status} {engine.title()}")
        
        print(f"\nTotal: {len(available)}/{len(all_engines)} engines available")
    
    def health_check(self):
        """Perform system health check."""
        # Create basic health check without requiring full imports
        try:
            if CRAWLER_AVAILABLE:
                try:
                    # Try to import health check function
                    sys.path.append(str(Path(__file__).parent))
                    import __init__ as search_crawler_init
                    health = search_crawler_init.get_search_crawler_health()
                except ImportError:
                    # Fallback health check
                    health = {
                        'status': 'degraded',
                        'core_available': CRAWLER_AVAILABLE,
                        'engines_available': bool(SEARCH_ENGINES),
                        'analysis_available': CRAWLER_AVAILABLE,
                        'config_available': CRAWLER_AVAILABLE,
                        'crawlee_enhanced_available': CRAWLER_AVAILABLE and CRAWLEE_AVAILABLE,
                        'crawlee_library_available': CRAWLEE_AVAILABLE,
                        'version': '1.0.0'
                    }
            else:
                health = {
                    'status': 'degraded',
                    'core_available': False,
                    'engines_available': False,
                    'analysis_available': False,
                    'config_available': False,
                    'crawlee_enhanced_available': False,
                    'crawlee_library_available': False,
                    'version': '1.0.0'
                }
        except Exception as e:
            self.print_error(f"Health check failed: {e}")
            return
        
        if self.console:
            # Create health status table
            table = Table(title="Search Crawler Health Check")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="dim")
            
            components = [
                ("Core Components", "✅ Available" if health['core_available'] else "❌ Unavailable", "SearchCrawler, ConflictSearchCrawler"),
                ("Search Engines", "✅ Available" if health['engines_available'] else "❌ Unavailable", f"{len(get_available_engines())} engines"),
                ("Analysis Tools", "✅ Available" if health['analysis_available'] else "❌ Unavailable", "Keywords, Conflict Detection"),
                ("Crawlee Integration", "✅ Available" if health['crawlee_enhanced_available'] else "❌ Unavailable", f"Library: {health['crawlee_library_available']}"),
                ("Configuration", "✅ Available" if health['config_available'] else "❌ Unavailable", "Config management"),
                ("CLI Enhancements", "✅ Available" if CLI_ENHANCED else "❌ Limited", "Rich UI components")
            ]
            
            for component, status, details in components:
                table.add_row(component, status, details)
            
            self.console.print(table)
            
            # Overall status
            overall_status = "🟢 Healthy" if health['status'] == 'healthy' else "🟡 Degraded"
            self.console.print(f"\nOverall Status: {overall_status}")
            self.console.print(f"Version: {health['version']}")
        else:
            print("Search Crawler Health Check")
            print("=" * 30)
            for key, value in health.items():
                print(f"{key}: {value}")
    
    async def create_crawler(self):
        """Create appropriate crawler based on configuration."""
        try:
            if self.config.mode == "crawlee" and CRAWLEE_AVAILABLE:
                # Create Crawlee-enhanced crawler
                config = create_crawlee_search_config(
                    engines=self.config.engines,
                    max_results=self.config.max_results,
                    enable_content_extraction=self.config.extract_content,
                    target_countries=self.config.target_countries,
                    preferred_extraction_method=self.config.preferred_extraction_method,
                    min_content_length=self.config.min_content_length,
                    min_quality_score=self.config.min_quality_score
                )
                
                from core.crawlee_enhanced_search_crawler import create_crawlee_search_crawler
                self.crawler = await create_crawlee_search_crawler(config)
                
            elif self.config.mode == "conflict":
                # Create conflict monitoring crawler
                config = ConflictSearchConfig(
                    engines=self.config.engines,
                    max_results_per_engine=self.config.max_results_per_engine,
                    total_max_results=self.config.max_results,
                    download_content=self.config.download_content,
                    enable_alerts=self.config.enable_alerts,
                    escalation_threshold=self.config.escalation_threshold,
                    timeout=self.config.timeout,
                    use_stealth=self.config.enable_stealth,
                    conflict_regions=self.config.conflict_regions
                )
                
                self.crawler = ConflictSearchCrawler(config)
                
            else:
                # Create general search crawler
                config = SearchCrawlerConfig(
                    engines=self.config.engines,
                    max_results_per_engine=self.config.max_results_per_engine,
                    total_max_results=self.config.max_results,
                    download_content=self.config.download_content,
                    timeout=self.config.timeout,
                    use_stealth=self.config.enable_stealth
                )
                
                self.crawler = SearchCrawler(config)
            
            self.print_success(f"Created {self.config.mode} crawler with {len(self.config.engines)} engines")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create crawler: {e}")
            return False
    
    async def perform_search(self) -> bool:
        """Perform search operation."""
        if not await self.create_crawler():
            return False
        
        try:
            self.start_time = time.time()
            
            if self.console and self.config.show_progress and not self.config.quiet:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task(f"Searching '{self.config.query}'...", total=None)
                    
                    if hasattr(self.crawler, 'search_with_content'):
                        # Crawlee-enhanced search
                        self.results = await self.crawler.search_with_content(
                            query=self.config.query,
                            max_results=self.config.max_results,
                            extract_content=self.config.extract_content
                        )
                    elif hasattr(self.crawler, 'search_conflicts'):
                        # Conflict search
                        self.results = await self.crawler.search_conflicts(
                            region='horn_of_africa',
                            keywords=[self.config.query],
                            max_results=self.config.max_results
                        )
                    else:
                        # General search
                        self.results = await self.crawler.search(
                            query=self.config.query,
                            max_results=self.config.max_results
                        )
                    
                    progress.update(task, completed=True)
            else:
                # Search without progress bar
                if hasattr(self.crawler, 'search_with_content'):
                    self.results = await self.crawler.search_with_content(
                        query=self.config.query,
                        max_results=self.config.max_results,
                        extract_content=self.config.extract_content
                    )
                elif hasattr(self.crawler, 'search_conflicts'):
                    self.results = await self.crawler.search_conflicts(
                        region='horn_of_africa',
                        keywords=[self.config.query],
                        max_results=self.config.max_results
                    )
                else:
                    self.results = await self.crawler.search(
                        query=self.config.query,
                        max_results=self.config.max_results
                    )
            
            # Apply post-processing
            if self.config.deduplicate_results:
                self.results = self.deduplicate_results(self.results)
            
            if self.config.enable_quality_filtering:
                self.results = self.filter_by_quality(self.results)
            
            elapsed = time.time() - self.start_time
            self.print_success(f"Found {len(self.results)} results in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            self.print_error(f"Search failed: {e}")
            return False
        
        finally:
            if self.crawler and hasattr(self.crawler, 'close'):
                await self.crawler.close()
    
    def deduplicate_results(self, results):
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        if len(results) != len(unique_results):
            self.print_message(f"Removed {len(results) - len(unique_results)} duplicate results")
        
        return unique_results
    
    def filter_by_quality(self, results):
        """Filter results by quality score."""
        if not hasattr(results[0], 'content_quality_score') if results else False:
            return results
        
        filtered_results = [
            r for r in results 
            if getattr(r, 'content_quality_score', 1.0) >= self.config.min_quality_score
        ]
        
        if len(results) != len(filtered_results):
            self.print_message(f"Filtered out {len(results) - len(filtered_results)} low-quality results")
        
        return filtered_results
    
    def format_output(self):
        """Format and display results."""
        if not self.results:
            self.print_warning("No results to display")
            return
        
        if self.config.output_format == "json":
            self.output_json()
        elif self.config.output_format == "csv":
            self.output_csv()
        elif self.config.output_format == "yaml":
            self.output_yaml()
        elif self.config.output_format == "table":
            self.output_table()
        else:
            self.output_text()
    
    def output_json(self):
        """Output results in JSON format."""
        results_data = []
        for result in self.results:
            if hasattr(result, '__dict__'):
                result_dict = {k: v for k, v in result.__dict__.items() 
                             if not k.startswith('_') and v is not None}
            else:
                result_dict = {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'engine': result.engine
                }
            results_data.append(result_dict)
        
        output = {
            'query': self.config.query,
            'mode': self.config.mode,
            'engines': self.config.engines,
            'total_results': len(self.results),
            'search_time': time.time() - self.start_time if self.start_time else 0,
            'results': results_data
        }
        
        json_str = json.dumps(output, indent=2, default=str)
        
        if self.config.output_file:
            with open(self.config.output_file, 'w') as f:
                f.write(json_str)
            self.print_success(f"Results saved to {self.config.output_file}")
        else:
            print(json_str)
    
    def output_csv(self):
        """Output results in CSV format."""
        if not self.results:
            return
        
        fieldnames = ['title', 'url', 'snippet', 'engine', 'rank']
        
        # Add additional fields if available
        first_result = self.results[0]
        if hasattr(first_result, 'conflict_score'):
            fieldnames.append('conflict_score')
        if hasattr(first_result, 'content_quality_score'):
            fieldnames.append('content_quality_score')
        if hasattr(first_result, 'timestamp'):
            fieldnames.append('timestamp')
        
        output_file = self.config.output_file or sys.stdout
        
        if isinstance(output_file, str):
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                self._write_csv_rows(writer, fieldnames)
            self.print_success(f"Results saved to {output_file}")
        else:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            self._write_csv_rows(writer, fieldnames)
    
    def _write_csv_rows(self, writer, fieldnames):
        """Write CSV rows."""
        for result in self.results:
            row = {}
            for field in fieldnames:
                value = getattr(result, field, '')
                if isinstance(value, (list, dict)):
                    value = str(value)
                row[field] = value
            writer.writerow(row)
    
    def output_yaml(self):
        """Output results in YAML format."""
        results_data = []
        for result in self.results:
            if hasattr(result, '__dict__'):
                result_dict = {k: v for k, v in result.__dict__.items() 
                             if not k.startswith('_') and v is not None}
            else:
                result_dict = {
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'engine': result.engine
                }
            results_data.append(result_dict)
        
        output = {
            'query': self.config.query,
            'mode': self.config.mode,
            'engines': self.config.engines,
            'total_results': len(self.results),
            'results': results_data
        }
        
        yaml_str = yaml.dump(output, default_flow_style=False, indent=2)
        
        if self.config.output_file:
            with open(self.config.output_file, 'w') as f:
                f.write(yaml_str)
            self.print_success(f"Results saved to {self.config.output_file}")
        else:
            print(yaml_str)
    
    def output_table(self):
        """Output results in table format."""
        if not self.console:
            self.output_text()
            return
        
        table = Table(title=f"Search Results: '{self.config.query}'")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", max_width=50)
        table.add_column("Engine", style="green", width=10)
        table.add_column("URL", style="blue", max_width=40)
        
        # Add optional columns
        if self.results and hasattr(self.results[0], 'conflict_score'):
            table.add_column("Conflict", style="red", width=8)
        if self.results and hasattr(self.results[0], 'content_quality_score'):
            table.add_column("Quality", style="yellow", width=8)
        
        for i, result in enumerate(self.results[:50], 1):  # Limit to 50 for display
            row = [
                str(i),
                result.title[:47] + "..." if len(result.title) > 50 else result.title,
                result.engine.upper(),
                result.url[:37] + "..." if len(result.url) > 40 else result.url
            ]
            
            if hasattr(result, 'conflict_score'):
                row.append(f"{result.conflict_score:.2f}")
            if hasattr(result, 'content_quality_score'):
                row.append(f"{result.content_quality_score:.2f}")
            
            table.add_row(*row)
        
        self.console.print(table)
        
        if len(self.results) > 50:
            self.console.print(f"\n[dim]Showing first 50 of {len(self.results)} results[/dim]")
        
        # Summary statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.console.print(f"\n[bold]Summary:[/bold] {len(self.results)} results in {elapsed:.2f}s")
    
    def output_text(self):
        """Output results in plain text format."""
        print(f"\nSearch Results for: '{self.config.query}'")
        print("=" * 60)
        print(f"Mode: {self.config.mode.upper()}")
        print(f"Engines: {', '.join(self.config.engines)}")
        print(f"Total Results: {len(self.results)}")
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Search Time: {elapsed:.2f}s")
        
        print("\nResults:")
        print("-" * 40)
        
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   Engine: {result.engine.upper()}")
            print(f"   URL: {result.url}")
            
            if hasattr(result, 'snippet') and result.snippet:
                snippet = result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet
                print(f"   Snippet: {snippet}")
            
            if hasattr(result, 'conflict_score'):
                print(f"   Conflict Score: {result.conflict_score:.3f}")
            
            if hasattr(result, 'content_quality_score'):
                print(f"   Quality Score: {result.content_quality_score:.3f}")
    
    def interactive_mode(self):
        """Launch interactive configuration mode."""
        if not CLI_ENHANCED:
            self.print_error("Interactive mode requires 'rich' library. Install with: pip install rich")
            return
        
        self.console.print(Panel.fit("🔍 Search Crawler Interactive Mode", style="bold blue"))
        
        # Query input
        query = Prompt.ask("Enter search query")
        self.config.query = query
        
        # Mode selection
        mode = Prompt.ask(
            "Select crawler mode",
            choices=["general", "conflict", "crawlee", "auto"],
            default="auto"
        )
        self.config.mode = mode
        
        # Engine selection
        if Confirm.ask("Customize search engines?", default=False):
            available = get_available_engines() if CRAWLER_AVAILABLE else ["duckduckgo"]
            selected = []
            for engine in available:
                if Confirm.ask(f"Use {engine.title()}?", default=engine in ["google", "bing", "duckduckgo"]):
                    selected.append(engine)
            self.config.engines = selected
        
        # Results configuration
        self.config.max_results = IntPrompt.ask("Maximum results", default=50)
        self.config.download_content = Confirm.ask("Download full content?", default=False)
        
        if mode == "conflict":
            self.config.enable_alerts = Confirm.ask("Enable conflict alerts?", default=True)
        
        # Output format
        output_format = Prompt.ask(
            "Output format",
            choices=["table", "json", "csv", "text"],
            default="table"
        )
        self.config.output_format = output_format
        
        # Save configuration
        if Confirm.ask("Save this configuration?", default=False):
            config_file = Prompt.ask("Configuration file", default="search_config.yaml")
            self.save_configuration(config_file)
        
        self.console.print("\n[bold green]Configuration complete![/bold green]")
        
        # Run search
        if Confirm.ask("Run search now?", default=True):
            asyncio.run(self.run_search())
    
    async def run_search(self):
        """Main search execution."""
        if await self.perform_search():
            self.format_output()
            
            # Save raw results if requested
            if hasattr(self.config, 'save_raw') and self.config.save_raw:
                with open(self.config.save_raw, 'w') as f:
                    json.dump([r.__dict__ for r in self.results], f, indent=2, default=str)
            
            # Save configuration if requested
            if self.config.save_config:
                self.save_configuration(self.config.save_config)
    
    async def run(self):
        """Main CLI entry point."""
        try:
            args = self.parse_arguments()
            
            # Handle benchmark
            if hasattr(args, 'benchmark') and args.benchmark:
                await self.run_benchmark()
                return
            
            # Handle batch mode
            if self.config.batch_mode:
                await self.run_batch_mode()
                return
            
            # Handle monitor mode
            if self.config.monitor_mode:
                await self.run_monitor_mode(getattr(args, 'monitor_interval', 300))
                return
            
            # Regular search
            await self.run_search()
            
        except KeyboardInterrupt:
            self.print_message("\nOperation cancelled by user", "yellow")
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            if self.config.verbose:
                import traceback
                traceback.print_exc()
    
    async def run_benchmark(self):
        """Run performance benchmark."""
        self.print_message("Running Search Crawler benchmark...", "blue")
        
        # Import and run benchmark
        try:
            from tests.performance_benchmark import SearchCrawlerBenchmark
            benchmark = SearchCrawlerBenchmark()
            
            # Run subset of benchmarks
            await benchmark.benchmark_configuration_loading()
            await benchmark.benchmark_search_engine_initialization()
            await benchmark.benchmark_keyword_processing()
            
            report = benchmark.generate_performance_report()
            
            if self.console:
                self.console.print(Panel(report, title="Benchmark Results"))
            else:
                print(report)
                
        except ImportError:
            self.print_error("Benchmark module not available")
    
    async def run_batch_mode(self):
        """Run batch processing mode."""
        self.print_message("Batch mode not yet implemented", "yellow")
        # TODO: Implement batch processing from config file
    
    async def run_monitor_mode(self, interval: int):
        """Run continuous monitoring mode."""
        self.print_message(f"Starting monitor mode (interval: {interval}s)", "blue")
        
        try:
            while True:
                await self.run_search()
                self.print_message(f"Next check in {interval} seconds...", "dim")
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            self.print_message("Monitoring stopped", "yellow")


def main():
    """Main entry point."""
    cli = SearchCrawlerCLI()
    
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()