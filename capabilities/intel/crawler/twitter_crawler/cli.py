#!/usr/bin/env python3
"""
Twitter Crawler CLI
===================

Comprehensive command-line interface for the Twitter Crawler package with 
intelligent defaults, performance optimization, and advanced monitoring capabilities.

Features:
- Multi-mode operation (search, monitor, analyze, stream)
- Performance-optimized operations with connection pooling
- Real-time monitoring and conflict detection
- Advanced analytics and reporting
- Batch processing and automation
- Export capabilities (JSON, CSV, Excel)
- Interactive configuration wizard

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
import signal
import threading

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# CLI-specific imports
try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Confirm, Prompt, IntPrompt
    from rich.layout import Layout
    from rich.text import Text
    CLI_ENHANCED = True
except ImportError:
    CLI_ENHANCED = False
    Console = None

# Twitter crawler imports
try:
    from optimized_core import (
        OptimizedTwitterCrawler, OptimizedTwitterConfig, PerformanceMetrics,
        CrawlerStatus, create_optimized_twitter_crawler
    )
    from core import TwitterCrawler, TwitterConfig
    from search import TwitterSearchEngine, SearchQuery, QueryBuilder, ConflictSearchTemplates
    from monitoring import TwitterMonitor, ConflictMonitor, AlertLevel
    from analysis import TwitterAnalyzer, SentimentAnalyzer
    CRAWLER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Twitter crawler components not available: {e}")
    CRAWLER_AVAILABLE = False


class OperationMode(Enum):
    """Available operation modes"""
    SEARCH = "search"
    MONITOR = "monitor"
    ANALYZE = "analyze"
    STREAM = "stream"
    BATCH = "batch"
    INTERACTIVE = "interactive"


class OutputFormat(Enum):
    """Available output formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    TEXT = "text"
    TABLE = "table"
    YAML = "yaml"


@dataclass
class CLIConfig:
    """CLI configuration with intelligent defaults"""
    
    # Core operation settings
    mode: str = "search"
    query: str = ""
    username: str = ""
    password: str = ""
    email: str = ""
    
    # Search parameters
    max_tweets: int = 100
    result_type: str = "recent"  # recent, popular, mixed
    language: str = "en"
    include_retweets: bool = True
    include_replies: bool = True
    verified_only: bool = False
    
    # Geographic and temporal filters
    location: str = ""
    radius: str = "10km"
    since_date: str = ""
    until_date: str = ""
    
    # Monitoring settings
    keywords: List[str] = None
    hashtags: List[str] = None
    users_to_monitor: List[str] = None
    alert_threshold: int = 10
    monitor_interval: int = 300  # 5 minutes
    enable_alerts: bool = False
    
    # Performance settings
    use_optimized: bool = True
    max_concurrent: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600
    connection_pool_size: int = 20
    rate_limit_per_minute: int = 30
    
    # Analysis settings
    enable_sentiment: bool = False
    enable_conflict_detection: bool = False
    confidence_threshold: float = 0.7
    
    # Output settings
    output_format: str = "table"
    output_file: str = ""
    export_user_data: bool = True
    export_metadata: bool = True
    pretty_print: bool = True
    
    # Advanced settings
    verbose: bool = False
    quiet: bool = False
    debug: bool = False
    show_progress: bool = True
    save_config: str = ""
    load_config: str = ""
    
    # Session management
    save_session: bool = True
    session_file: str = "twitter_session.pkl"
    
    def __post_init__(self):
        """Apply intelligent defaults"""
        if self.keywords is None:
            self.keywords = []
        if self.hashtags is None:
            self.hashtags = []
        if self.users_to_monitor is None:
            self.users_to_monitor = []
        
        # Adjust settings based on mode
        if self.mode == "monitor":
            self.show_progress = False  # Don't show progress for monitoring
            self.enable_alerts = True
        elif self.mode == "analyze":
            self.enable_sentiment = True
            self.export_metadata = True
        elif self.mode == "search" and "conflict" in self.query.lower():
            self.enable_conflict_detection = True


class TwitterCrawlerCLI:
    """Comprehensive CLI for Twitter Crawler package"""
    
    def __init__(self):
        self.console = Console() if CLI_ENHANCED else None
        self.config = CLIConfig()
        self.crawler: Optional[Union[OptimizedTwitterCrawler, TwitterCrawler]] = None
        self.results = []
        self.start_time = None
        self.stop_monitoring = False
        
        # Setup logging
        self.setup_logging()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.config.debug else (
            logging.INFO if self.config.verbose else logging.WARNING
        )
        
        if self.config.quiet:
            log_level = logging.ERROR
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr) if not self.config.quiet else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.print_message("\nReceived shutdown signal, cleaning up...", "yellow")
        self.stop_monitoring = True
        if self.crawler:
            asyncio.create_task(self.crawler.close())
        sys.exit(0)
    
    def print_message(self, message: str, style: str = ""):
        """Print message with optional styling"""
        if self.config.quiet:
            return
        
        if self.console and style:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_error(self, message: str):
        """Print error message"""
        if self.console:
            self.console.print(f"❌ Error: {message}", style="bold red")
        else:
            print(f"Error: {message}", file=sys.stderr)
    
    def print_warning(self, message: str):
        """Print warning message"""
        if not self.config.quiet:
            if self.console:
                self.console.print(f"⚠️  Warning: {message}", style="bold yellow")
            else:
                print(f"Warning: {message}")
    
    def print_success(self, message: str):
        """Print success message"""
        if self.console:
            self.console.print(f"✅ {message}", style="bold green")
        else:
            print(f"Success: {message}")
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            description="Twitter Crawler CLI - Advanced Twitter data collection and analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic tweet search
  %(prog)s search "machine learning" --max-tweets 50

  # Conflict monitoring
  %(prog)s monitor --keywords "Ethiopia conflict,Somalia violence" --enable-alerts

  # User analysis
  %(prog)s analyze --users "elonmusk,twitter" --enable-sentiment

  # Geographic search
  %(prog)s search "earthquake" --location "San Francisco" --radius "50km"

  # Batch processing
  %(prog)s batch --load-config batch_config.yaml

  # Interactive mode
  %(prog)s interactive

  # Real-time monitoring with alerts
  %(prog)s monitor --keywords "breaking news,alert" --alert-threshold 5 \\
    --monitor-interval 60 --output-format json --output-file alerts.json

For more information: https://github.com/your-repo/twitter-crawler
            """
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
        
        # Search command
        search_parser = subparsers.add_parser('search', help='Search for tweets')
        self.add_search_arguments(search_parser)
        
        # Monitor command
        monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
        self.add_monitor_arguments(monitor_parser)
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze tweets and users')
        self.add_analyze_arguments(analyze_parser)
        
        # Stream command
        stream_parser = subparsers.add_parser('stream', help='Real-time tweet streaming')
        self.add_stream_arguments(stream_parser)
        
        # Batch command
        batch_parser = subparsers.add_parser('batch', help='Batch processing')
        self.add_batch_arguments(batch_parser)
        
        # Interactive command
        interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
        
        # Global options
        for subparser in [search_parser, monitor_parser, analyze_parser, stream_parser, batch_parser]:
            self.add_global_arguments(subparser)
        
        # Utility commands
        parser.add_argument('--health-check', action='store_true', help='Check system health')
        parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
        parser.add_argument('--version', action='version', version='Twitter Crawler CLI v2.0.0')
        
        return parser
    
    def add_search_arguments(self, parser):
        """Add search-specific arguments"""
        parser.add_argument('query', nargs='?', help='Search query')
        parser.add_argument('--max-tweets', type=int, default=100, help='Maximum tweets to fetch')
        parser.add_argument('--result-type', choices=['recent', 'popular', 'mixed'], 
                          default='recent', help='Type of results')
        parser.add_argument('--language', default='en', help='Language code (e.g., en, es, fr)')
        parser.add_argument('--location', help='Location filter (e.g., "New York")')
        parser.add_argument('--radius', default='10km', help='Search radius (e.g., 10km, 5mi)')
        parser.add_argument('--since-date', help='Start date (YYYY-MM-DD)')
        parser.add_argument('--until-date', help='End date (YYYY-MM-DD)')
        parser.add_argument('--verified-only', action='store_true', help='Only verified users')
        parser.add_argument('--exclude-retweets', action='store_true', help='Exclude retweets')
        parser.add_argument('--exclude-replies', action='store_true', help='Exclude replies')
    
    def add_monitor_arguments(self, parser):
        """Add monitoring-specific arguments"""
        parser.add_argument('--keywords', help='Comma-separated keywords to monitor')
        parser.add_argument('--hashtags', help='Comma-separated hashtags to monitor')
        parser.add_argument('--users', help='Comma-separated usernames to monitor')
        parser.add_argument('--alert-threshold', type=int, default=10, 
                          help='Number of tweets to trigger alert')
        parser.add_argument('--monitor-interval', type=int, default=300,
                          help='Monitoring interval in seconds')
        parser.add_argument('--enable-alerts', action='store_true', help='Enable alert system')
        parser.add_argument('--enable-conflict-detection', action='store_true',
                          help='Enable conflict detection')
    
    def add_analyze_arguments(self, parser):
        """Add analysis-specific arguments"""
        parser.add_argument('--enable-sentiment', action='store_true', help='Enable sentiment analysis')
        parser.add_argument('--confidence-threshold', type=float, default=0.7,
                          help='Confidence threshold for analysis')
        parser.add_argument('--analyze-users', help='Comma-separated usernames to analyze')
        parser.add_argument('--analyze-hashtags', help='Comma-separated hashtags to analyze')
        parser.add_argument('--trend-analysis', action='store_true', help='Enable trend analysis')
    
    def add_stream_arguments(self, parser):
        """Add streaming-specific arguments"""
        parser.add_argument('--stream-keywords', help='Keywords for streaming')
        parser.add_argument('--stream-users', help='Users to stream')
        parser.add_argument('--stream-duration', type=int, default=3600,
                          help='Streaming duration in seconds')
        parser.add_argument('--buffer-size', type=int, default=1000, help='Stream buffer size')
    
    def add_batch_arguments(self, parser):
        """Add batch processing arguments"""
        parser.add_argument('--batch-file', help='Batch configuration file')
        parser.add_argument('--batch-queries', help='File with queries (one per line)')
        parser.add_argument('--batch-users', help='File with usernames (one per line)')
        parser.add_argument('--parallel-jobs', type=int, default=5, help='Number of parallel jobs')
    
    def add_global_arguments(self, parser):
        """Add global arguments to all subcommands"""
        # Authentication
        auth_group = parser.add_argument_group('Authentication')
        auth_group.add_argument('--username', help='Twitter username')
        auth_group.add_argument('--password', help='Twitter password') 
        auth_group.add_argument('--email', help='Twitter email')
        
        # Performance
        perf_group = parser.add_argument_group('Performance')
        perf_group.add_argument('--use-optimized', action='store_true', default=True,
                               help='Use optimized crawler (default: true)')
        perf_group.add_argument('--max-concurrent', type=int, default=10,
                               help='Maximum concurrent requests')
        perf_group.add_argument('--enable-caching', action='store_true', default=True,
                               help='Enable caching (default: true)')
        perf_group.add_argument('--cache-ttl', type=int, default=3600,
                               help='Cache TTL in seconds')
        perf_group.add_argument('--connection-pool-size', type=int, default=20,
                               help='Connection pool size')
        
        # Output
        output_group = parser.add_argument_group('Output')
        output_group.add_argument('--output-format', choices=[f.value for f in OutputFormat],
                                 default='table', help='Output format')
        output_group.add_argument('--output-file', help='Output file path')
        output_group.add_argument('--export-user-data', action='store_true', default=True,
                                 help='Export user data (default: true)')
        output_group.add_argument('--export-metadata', action='store_true', default=True,
                                 help='Export metadata (default: true)')
        output_group.add_argument('--pretty-print', action='store_true', default=True,
                                 help='Pretty print output (default: true)')
        
        # Configuration
        config_group = parser.add_argument_group('Configuration')
        config_group.add_argument('--save-config', help='Save configuration to file')
        config_group.add_argument('--load-config', help='Load configuration from file')
        config_group.add_argument('--save-session', action='store_true', default=True,
                                 help='Save session (default: true)')
        config_group.add_argument('--session-file', default='twitter_session.pkl',
                                 help='Session file path')
        
        # Logging
        log_group = parser.add_argument_group('Logging')
        log_group.add_argument('--verbose', action='store_true', help='Verbose output')
        log_group.add_argument('--quiet', action='store_true', help='Suppress output')
        log_group.add_argument('--debug', action='store_true', help='Debug mode')
        log_group.add_argument('--show-progress', action='store_true', default=True,
                              help='Show progress bar (default: true)')
    
    def parse_arguments(self, args=None):
        """Parse command line arguments"""
        parser = self.create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Handle utility commands
        if parsed_args.health_check:
            self.health_check()
            sys.exit(0)
        
        if parsed_args.benchmark:
            asyncio.run(self.run_benchmark())
            sys.exit(0)
        
        # Load configuration if specified
        if hasattr(parsed_args, 'load_config') and parsed_args.load_config:
            if not self.load_configuration(parsed_args.load_config):
                sys.exit(1)
        
        # Update configuration from arguments
        self.update_config_from_args(parsed_args)
        
        # Validate configuration
        if not self.validate_configuration():
            sys.exit(1)
        
        return parsed_args
    
    def update_config_from_args(self, args):
        """Update configuration from parsed arguments"""
        # Core settings
        if hasattr(args, 'mode') and args.mode:
            self.config.mode = args.mode
        
        if hasattr(args, 'query') and args.query:
            self.config.query = args.query
        
        # Authentication
        for attr in ['username', 'password', 'email']:
            if hasattr(args, attr) and getattr(args, attr):
                setattr(self.config, attr, getattr(args, attr))
        
        # Search parameters
        search_params = [
            'max_tweets', 'result_type', 'language', 'location', 'radius',
            'since_date', 'until_date', 'verified_only'
        ]
        for param in search_params:
            if hasattr(args, param.replace('_', '-')) or hasattr(args, param):
                value = getattr(args, param.replace('_', '-'), None) or getattr(args, param, None)
                if value is not None:
                    setattr(self.config, param, value)
        
        # Handle boolean flags
        if hasattr(args, 'exclude_retweets') and args.exclude_retweets:
            self.config.include_retweets = False
        if hasattr(args, 'exclude_replies') and args.exclude_replies:
            self.config.include_replies = False
        
        # Monitoring settings
        if hasattr(args, 'keywords') and args.keywords:
            self.config.keywords = [k.strip() for k in args.keywords.split(',')]
        if hasattr(args, 'hashtags') and args.hashtags:
            self.config.hashtags = [h.strip() for h in args.hashtags.split(',')]
        if hasattr(args, 'users') and args.users:
            self.config.users_to_monitor = [u.strip() for u in args.users.split(',')]
        
        # Performance settings
        perf_params = [
            'use_optimized', 'max_concurrent', 'enable_caching', 'cache_ttl',
            'connection_pool_size', 'alert_threshold', 'monitor_interval'
        ]
        for param in perf_params:
            if hasattr(args, param.replace('_', '-')) or hasattr(args, param):
                value = getattr(args, param.replace('_', '-'), None) or getattr(args, param, None)
                if value is not None:
                    setattr(self.config, param, value)
        
        # Analysis settings
        analysis_params = ['enable_sentiment', 'enable_conflict_detection', 'confidence_threshold']
        for param in analysis_params:
            if hasattr(args, param.replace('_', '-')) or hasattr(args, param):
                value = getattr(args, param.replace('_', '-'), None) or getattr(args, param, None)
                if value is not None:
                    setattr(self.config, param, value)
        
        # Output settings
        output_params = [
            'output_format', 'output_file', 'export_user_data', 'export_metadata', 'pretty_print'
        ]
        for param in output_params:
            if hasattr(args, param.replace('_', '-')) or hasattr(args, param):
                value = getattr(args, param.replace('_', '-'), None) or getattr(args, param, None)
                if value is not None:
                    setattr(self.config, param, value)
        
        # Logging settings
        log_params = ['verbose', 'quiet', 'debug', 'show_progress']
        for param in log_params:
            if hasattr(args, param) and getattr(args, param):
                setattr(self.config, param, True)
        
        # Configuration management
        if hasattr(args, 'save_config') and args.save_config:
            self.config.save_config = args.save_config
        if hasattr(args, 'save_session') and args.save_session:
            self.config.save_session = args.save_session
        if hasattr(args, 'session_file') and args.session_file:
            self.config.session_file = args.session_file
        
        # Re-apply intelligent defaults
        self.config.__post_init__()
    
    def validate_configuration(self) -> bool:
        """Validate configuration parameters"""
        if not CRAWLER_AVAILABLE:
            self.print_error("Twitter crawler components are not available")
            return False
        
        if self.config.mode in ["search"] and not self.config.query:
            self.print_error("Search query is required for search mode")
            return False
        
        if self.config.mode == "monitor" and not any([
            self.config.keywords, self.config.hashtags, self.config.users_to_monitor
        ]):
            self.print_error("Keywords, hashtags, or users are required for monitoring mode")
            return False
        
        if self.config.max_tweets <= 0:
            self.print_error("Max tweets must be positive")
            return False
        
        if self.config.confidence_threshold < 0 or self.config.confidence_threshold > 1:
            self.print_error("Confidence threshold must be between 0 and 1")
            return False
        
        return True
    
    def load_configuration(self, config_file: str) -> bool:
        """Load configuration from file"""
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
    
    def save_configuration(self, config_file: str) -> bool:
        """Save current configuration to file"""
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
    
    def health_check(self):
        """Perform system health check"""
        if not CRAWLER_AVAILABLE:
            self.print_error("Twitter crawler components not available")
            return
        
        health_info = {
            "twitter_crawler_available": CRAWLER_AVAILABLE,
            "cli_enhanced": CLI_ENHANCED,
            "optimized_crawler": self.config.use_optimized,
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        if self.console:
            table = Table(title="Twitter Crawler Health Check")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="dim")
            
            for component, status in health_info.items():
                if isinstance(status, bool):
                    status_text = "✅ Available" if status else "❌ Unavailable"
                else:
                    status_text = str(status)
                
                table.add_row(component.replace('_', ' ').title(), status_text, "")
            
            self.console.print(table)
        else:
            print("Twitter Crawler Health Check")
            print("=" * 30)
            for key, value in health_info.items():
                print(f"{key}: {value}")
    
    async def create_crawler(self):
        """Create appropriate crawler based on configuration"""
        try:
            if self.config.use_optimized:
                # Create optimized crawler
                crawler_config = OptimizedTwitterConfig(
                    username=self.config.username,
                    password=self.config.password,
                    email=self.config.email,
                    session_file=self.config.session_file,
                    auto_save_session=self.config.save_session,
                    max_concurrent_requests=self.config.max_concurrent,
                    enable_caching=self.config.enable_caching,
                    cache_ttl=self.config.cache_ttl,
                    connection_pool_size=self.config.connection_pool_size,
                    rate_limit_requests_per_minute=self.config.rate_limit_per_minute,
                    log_level="DEBUG" if self.config.debug else (
                        "INFO" if self.config.verbose else "WARNING"
                    ),
                    enable_metrics=True,
                    enable_health_checks=True
                )
                
                self.crawler = OptimizedTwitterCrawler(crawler_config)
            else:
                # Create standard crawler
                crawler_config = TwitterConfig(
                    username=self.config.username,
                    password=self.config.password,
                    email=self.config.email,
                    session_file=self.config.session_file,
                    auto_save_session=self.config.save_session,
                    max_concurrent_requests=self.config.max_concurrent,
                    log_level="DEBUG" if self.config.debug else (
                        "INFO" if self.config.verbose else "WARNING"
                    )
                )
                
                self.crawler = TwitterCrawler(crawler_config)
            
            # Initialize crawler
            success = await self.crawler.initialize()
            if success:
                crawler_type = "optimized" if self.config.use_optimized else "standard"
                self.print_success(f"Created {crawler_type} Twitter crawler")
                return True
            else:
                self.print_error("Failed to initialize crawler")
                return False
                
        except Exception as e:
            self.print_error(f"Failed to create crawler: {e}")
            return False
    
    async def run_search(self) -> bool:
        """Run search operation"""
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
                    
                    if hasattr(self.crawler, 'search_tweets'):
                        # Standard crawler
                        self.results = await self.crawler.search_tweets(
                            self.config.query,
                            count=self.config.max_tweets,
                            result_type=self.config.result_type
                        )
                    else:
                        # Optimized crawler
                        async def _search(client):
                            tweets = await client.search_tweet(
                                self.config.query, 
                                count=self.config.max_tweets
                            )
                            return [self.crawler._tweet_to_dict(tweet) for tweet in tweets]
                        
                        self.results = await self.crawler.make_optimized_request(_search)
                    
                    progress.update(task, completed=True)
            else:
                # Search without progress bar
                if hasattr(self.crawler, 'search_tweets'):
                    self.results = await self.crawler.search_tweets(
                        self.config.query,
                        count=self.config.max_tweets,
                        result_type=self.config.result_type
                    )
                else:
                    async def _search(client):
                        tweets = await client.search_tweet(
                            self.config.query, 
                            count=self.config.max_tweets
                        )
                        return [self.crawler._tweet_to_dict(tweet) for tweet in tweets]
                    
                    self.results = await self.crawler.make_optimized_request(_search)
            
            elapsed = time.time() - self.start_time
            self.print_success(f"Found {len(self.results)} tweets in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            self.print_error(f"Search failed: {e}")
            return False
        
        finally:
            if self.crawler:
                await self.crawler.close()
    
    async def run_monitoring(self) -> bool:
        """Run monitoring operation"""
        if not await self.create_crawler():
            return False
        
        try:
            self.print_message(f"Starting monitoring mode...", "blue")
            self.print_message(f"Keywords: {self.config.keywords}", "dim")
            self.print_message(f"Hashtags: {self.config.hashtags}", "dim")
            self.print_message(f"Users: {self.config.users_to_monitor}", "dim")
            self.print_message(f"Interval: {self.config.monitor_interval}s", "dim")
            
            iteration = 0
            total_tweets = 0
            
            while not self.stop_monitoring:
                iteration += 1
                iteration_start = time.time()
                
                self.print_message(f"\n🔍 Monitoring iteration {iteration}", "blue")
                
                # Build monitoring query
                query_parts = []
                query_parts.extend(self.config.keywords)
                query_parts.extend([f"#{tag}" for tag in self.config.hashtags])
                query_parts.extend([f"from:{user}" for user in self.config.users_to_monitor])
                
                if not query_parts:
                    self.print_error("No monitoring criteria specified")
                    break
                
                query = " OR ".join(query_parts)
                
                # Search for tweets
                try:
                    if hasattr(self.crawler, 'search_tweets'):
                        tweets = await self.crawler.search_tweets(
                            query,
                            count=100,
                            result_type='recent'
                        )
                    else:
                        async def _search(client):
                            tweets = await client.search_tweet(query, count=100)
                            return [self.crawler._tweet_to_dict(tweet) for tweet in tweets]
                        
                        tweets = await self.crawler.make_optimized_request(_search)
                    
                    total_tweets += len(tweets)
                    
                    # Check for alerts
                    if self.config.enable_alerts and len(tweets) >= self.config.alert_threshold:
                        self.print_message(
                            f"🚨 ALERT: {len(tweets)} tweets found (threshold: {self.config.alert_threshold})",
                            "bold red"
                        )
                        
                        # Save alerts if output file specified
                        if self.config.output_file:
                            alert_data = {
                                "timestamp": datetime.now().isoformat(),
                                "query": query,
                                "tweet_count": len(tweets),
                                "threshold": self.config.alert_threshold,
                                "tweets": tweets[:10]  # Save first 10 tweets
                            }
                            
                            with open(self.config.output_file, 'a') as f:
                                json.dump(alert_data, f)
                                f.write('\n')
                    
                    iteration_time = time.time() - iteration_start
                    self.print_message(
                        f"Found {len(tweets)} tweets in {iteration_time:.2f}s (Total: {total_tweets})",
                        "green"
                    )
                    
                except Exception as e:
                    self.print_error(f"Monitoring iteration {iteration} failed: {e}")
                
                # Wait for next iteration
                await asyncio.sleep(self.config.monitor_interval)
            
            return True
            
        except KeyboardInterrupt:
            self.print_message("Monitoring stopped by user", "yellow")
            return True
        except Exception as e:
            self.print_error(f"Monitoring failed: {e}")
            return False
        finally:
            if self.crawler:
                await self.crawler.close()
    
    async def run_analysis(self) -> bool:
        """Run analysis operation"""
        # Implementation would go here
        self.print_message("Analysis mode not yet implemented", "yellow")
        return True
    
    async def run_batch(self) -> bool:
        """Run batch operation"""
        # Implementation would go here
        self.print_message("Batch mode not yet implemented", "yellow")
        return True
    
    async def run_interactive(self) -> bool:
        """Run interactive mode"""
        if not CLI_ENHANCED:
            self.print_error("Interactive mode requires 'rich' library. Install with: pip install rich")
            return False
        
        self.console.print(Panel.fit("🐦 Twitter Crawler Interactive Mode", style="bold blue"))
        
        # Mode selection
        mode = Prompt.ask(
            "Select operation mode",
            choices=["search", "monitor", "analyze"],
            default="search"
        )
        self.config.mode = mode
        
        if mode == "search":
            # Search configuration
            query = Prompt.ask("Enter search query")
            self.config.query = query
            
            self.config.max_tweets = IntPrompt.ask("Maximum tweets", default=100)
            
            if Confirm.ask("Use advanced filters?", default=False):
                self.config.language = Prompt.ask("Language", default="en")
                self.config.result_type = Prompt.ask(
                    "Result type", 
                    choices=["recent", "popular", "mixed"],
                    default="recent"
                )
        
        elif mode == "monitor":
            # Monitoring configuration
            keywords = Prompt.ask("Enter keywords (comma-separated)", default="")
            if keywords:
                self.config.keywords = [k.strip() for k in keywords.split(',')]
            
            self.config.alert_threshold = IntPrompt.ask("Alert threshold", default=10)
            self.config.monitor_interval = IntPrompt.ask("Check interval (seconds)", default=300)
            self.config.enable_alerts = Confirm.ask("Enable alerts?", default=True)
        
        # Authentication
        if Confirm.ask("Provide authentication credentials?", default=False):
            self.config.username = Prompt.ask("Username")
            self.config.password = Prompt.ask("Password", password=True)
        
        # Output configuration
        self.config.output_format = Prompt.ask(
            "Output format",
            choices=["table", "json", "csv"],
            default="table"
        )
        
        if Confirm.ask("Save output to file?", default=False):
            self.config.output_file = Prompt.ask("Output file path")
        
        # Performance options
        self.config.use_optimized = Confirm.ask("Use optimized crawler?", default=True)
        
        self.console.print("\n[bold green]Configuration complete![/bold green]")
        
        # Run the configured operation
        if Confirm.ask("Run operation now?", default=True):
            return await self.run_operation()
        
        return True
    
    async def run_operation(self) -> bool:
        """Run the configured operation"""
        if self.config.mode == "search":
            return await self.run_search()
        elif self.config.mode == "monitor":
            return await self.run_monitoring()
        elif self.config.mode == "analyze":
            return await self.run_analysis()
        elif self.config.mode == "batch":
            return await self.run_batch()
        elif self.config.mode == "interactive":
            return await self.run_interactive()
        else:
            self.print_error(f"Unknown operation mode: {self.config.mode}")
            return False
    
    def format_output(self):
        """Format and display results"""
        if not self.results:
            self.print_warning("No results to display")
            return
        
        if self.config.output_format == "json":
            self.output_json()
        elif self.config.output_format == "csv":
            self.output_csv()
        elif self.config.output_format == "table":
            self.output_table()
        elif self.config.output_format == "yaml":
            self.output_yaml()
        else:
            self.output_text()
    
    def output_json(self):
        """Output results in JSON format"""
        output = {
            "query": self.config.query,
            "mode": self.config.mode,
            "total_results": len(self.results),
            "search_time": time.time() - self.start_time if self.start_time else 0,
            "config": asdict(self.config) if self.config.export_metadata else None,
            "results": self.results
        }
        
        json_str = json.dumps(output, indent=2 if self.config.pretty_print else None, default=str)
        
        if self.config.output_file:
            with open(self.config.output_file, 'w') as f:
                f.write(json_str)
            self.print_success(f"Results saved to {self.config.output_file}")
        else:
            print(json_str)
    
    def output_csv(self):
        """Output results in CSV format"""
        if not self.results:
            return
        
        fieldnames = ['id', 'text', 'created_at', 'user_username', 'retweet_count', 'favorite_count']
        
        if self.config.export_user_data:
            fieldnames.extend(['user_display_name', 'user_followers_count', 'user_verified'])
        
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
        """Write CSV rows"""
        for tweet in self.results:
            row = {}
            for field in fieldnames:
                if field.startswith('user_'):
                    user_field = field.replace('user_', '')
                    user_data = tweet.get('user', {})
                    value = user_data.get(user_field, '') if user_data else ''
                else:
                    value = tweet.get(field, '')
                
                if isinstance(value, (list, dict)):
                    value = str(value)
                row[field] = value
            writer.writerow(row)
    
    def output_table(self):
        """Output results in table format"""
        if not self.console:
            self.output_text()
            return
        
        table = Table(title=f"Twitter Search Results: '{self.config.query}'")
        table.add_column("#", style="dim", width=3)
        table.add_column("User", style="cyan", max_width=15)
        table.add_column("Tweet", style="white", max_width=60)
        table.add_column("Engagement", style="green", width=10)
        table.add_column("Date", style="blue", width=12)
        
        for i, tweet in enumerate(self.results[:50], 1):  # Limit to 50 for display
            user = tweet.get('user', {})
            username = user.get('username', 'Unknown')
            
            text = tweet.get('text', '')
            text = text[:57] + "..." if len(text) > 60 else text
            
            engagement = (
                tweet.get('retweet_count', 0) + 
                tweet.get('favorite_count', 0)
            )
            
            created_at = tweet.get('created_at', '')
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    date_str = date_obj.strftime('%Y-%m-%d')
                except:
                    date_str = created_at[:10]
            else:
                date_str = 'Unknown'
            
            table.add_row(
                str(i),
                f"@{username}",
                text,
                str(engagement),
                date_str
            )
        
        self.console.print(table)
        
        if len(self.results) > 50:
            self.console.print(f"\n[dim]Showing first 50 of {len(self.results)} results[/dim]")
        
        # Summary
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.console.print(f"\n[bold]Summary:[/bold] {len(self.results)} tweets in {elapsed:.2f}s")
    
    def output_text(self):
        """Output results in plain text format"""
        print(f"\nTwitter Search Results for: '{self.config.query}'")
        print("=" * 60)
        print(f"Total Results: {len(self.results)}")
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Search Time: {elapsed:.2f}s")
        
        print("\nTweets:")
        print("-" * 40)
        
        for i, tweet in enumerate(self.results, 1):
            user = tweet.get('user', {})
            username = user.get('username', 'Unknown')
            
            print(f"\n{i}. @{username}")
            print(f"   Text: {tweet.get('text', '')}")
            print(f"   Engagement: {tweet.get('retweet_count', 0)} RTs, {tweet.get('favorite_count', 0)} likes")
            print(f"   Date: {tweet.get('created_at', 'Unknown')}")
    
    def output_yaml(self):
        """Output results in YAML format"""
        output = {
            "query": self.config.query,
            "mode": self.config.mode,
            "total_results": len(self.results),
            "results": self.results
        }
        
        yaml_str = yaml.dump(output, default_flow_style=False, indent=2)
        
        if self.config.output_file:
            with open(self.config.output_file, 'w') as f:
                f.write(yaml_str)
            self.print_success(f"Results saved to {self.config.output_file}")
        else:
            print(yaml_str)
    
    async def run_benchmark(self):
        """Run performance benchmark"""
        self.print_message("Running Twitter Crawler benchmark...", "blue")
        
        benchmark_queries = [
            "machine learning",
            "climate change",
            "technology",
            "artificial intelligence",
            "social media"
        ]
        
        results = []
        
        for query in benchmark_queries:
            self.config.query = query
            self.config.max_tweets = 10
            
            start_time = time.time()
            success = await self.run_search()
            end_time = time.time()
            
            results.append({
                "query": query,
                "success": success,
                "duration": end_time - start_time,
                "tweets_found": len(self.results) if success else 0
            })
        
        # Display benchmark results
        if self.console:
            table = Table(title="Benchmark Results")
            table.add_column("Query", style="cyan")
            table.add_column("Duration", style="green")
            table.add_column("Tweets", style="yellow")
            table.add_column("Status", style="white")
            
            for result in results:
                status = "✅ Success" if result["success"] else "❌ Failed"
                table.add_row(
                    result["query"],
                    f"{result['duration']:.2f}s",
                    str(result["tweets_found"]),
                    status
                )
            
            self.console.print(table)
        else:
            print("Benchmark Results:")
            for result in results:
                print(f"Query: {result['query']}")
                print(f"Duration: {result['duration']:.2f}s")
                print(f"Tweets: {result['tweets_found']}")
                print(f"Status: {'Success' if result['success'] else 'Failed'}")
                print()
    
    async def run(self):
        """Main CLI entry point"""
        try:
            args = self.parse_arguments()
            
            # Handle specific modes
            if self.config.mode == "interactive":
                await self.run_interactive()
                return
            
            # Run the operation
            success = await self.run_operation()
            
            if success and self.results:
                self.format_output()
            
            # Save configuration if requested
            if self.config.save_config:
                self.save_configuration(self.config.save_config)
                
        except KeyboardInterrupt:
            self.print_message("\nOperation cancelled by user", "yellow")
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()


def main():
    """Main entry point"""
    cli = TwitterCrawlerCLI()
    
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()