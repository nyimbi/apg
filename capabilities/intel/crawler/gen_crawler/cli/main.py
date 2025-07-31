#!/usr/bin/env python3
"""
Gen Crawler CLI - Main Entry Point
==================================

Command-line interface for the generation crawler with comprehensive
configuration options and intelligent defaults.

Usage:
    gen-crawler crawl https://example.com --output ./results --format markdown
    gen-crawler analyze --config ./config.json --max-pages 100
    gen-crawler config --create --output ./gen-crawler-config.json

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import os
from datetime import datetime

from .utils import setup_logging, validate_urls, format_results
from ..config import create_gen_config, GenCrawlerSettings

logger = logging.getLogger(__name__)

def create_cli_parser() -> argparse.ArgumentParser:
    """Create the main CLI argument parser with comprehensive options."""
    
    parser = argparse.ArgumentParser(
        prog='gen-crawler',
        description='Next-generation web crawler using Crawlee AdaptivePlaywrightCrawler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic site crawl with markdown export
  gen-crawler crawl https://example.com --output ./results --format markdown
  
  # Advanced crawl with custom settings
  gen-crawler crawl https://news-site.com \\
    --max-pages 500 --max-concurrent 3 --crawl-delay 2.0 \\
    --include-patterns article,news,story --output ./news
  
  # Multiple sites with conflict monitoring
  gen-crawler crawl https://site1.com https://site2.com \\
    --conflict-keywords war,violence,crisis --format json
  
  # Create configuration file
  gen-crawler config --create --output ./my-config.json
  
  # Use existing configuration
  gen-crawler crawl https://example.com --config ./my-config.json
  
  # Export existing crawl data
  gen-crawler export ./crawl-results.json --format markdown --output ./markdown/
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Crawl command
    crawl_parser = subparsers.add_parser(
        'crawl', 
        help='Crawl one or more websites',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_crawl_arguments(crawl_parser)
    
    # Analyze command  
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze existing crawl data'
    )
    _add_analyze_arguments(analyze_parser)
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management'
    )
    _add_config_arguments(config_parser)
    
    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export crawl data to various formats'
    )
    _add_export_arguments(export_parser)
    
    # Global options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (use -v, -vv, or -vvv)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Write logs to file'
    )
    
    return parser

def _add_crawl_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the crawl command."""
    
    # Required arguments
    parser.add_argument(
        'urls',
        nargs='+',
        help='URLs to crawl (space-separated for multiple sites)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output', '-o',
        type=str,
        default='./crawl-results',
        help='Output directory (default: ./crawl-results)'
    )
    
    output_group.add_argument(
        '--format', '-f',
        choices=['markdown', 'json', 'csv', 'html'],
        default='json',
        help='Output format (default: json)'
    )
    
    output_group.add_argument(
        '--save-raw-html',
        action='store_true',
        help='Save raw HTML files alongside processed content'
    )
    
    output_group.add_argument(
        '--compress',
        action='store_true',
        help='Compress output files'
    )
    
    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--max-pages',
        type=int,
        default=500,
        help='Maximum pages per site (default: 500)'
    )
    
    perf_group.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent requests (default: 5)'
    )
    
    perf_group.add_argument(
        '--request-timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    perf_group.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries per request (default: 3)'
    )
    
    perf_group.add_argument(
        '--crawl-delay',
        type=float,
        default=2.0,
        help='Delay between requests in seconds (default: 2.0)'
    )
    
    perf_group.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Maximum crawl depth (default: 10)'
    )
    
    perf_group.add_argument(
        '--memory-limit',
        type=int,
        default=1024,
        help='Memory limit in MB (default: 1024)'
    )
    
    # Content filtering options
    filter_group = parser.add_argument_group('Content Filtering')
    filter_group.add_argument(
        '--min-content-length',
        type=int,
        default=100,
        help='Minimum content length in characters (default: 100)'
    )
    
    filter_group.add_argument(
        '--max-content-length',
        type=int,
        default=1000000,
        help='Maximum content length in characters (default: 1000000)'
    )
    
    filter_group.add_argument(
        '--include-patterns',
        type=str,
        help='Comma-separated patterns to include (e.g., article,news,story)'
    )
    
    filter_group.add_argument(
        '--exclude-patterns',
        type=str,
        help='Comma-separated patterns to exclude (e.g., tag,archive,login)'
    )
    
    filter_group.add_argument(
        '--exclude-extensions',
        type=str,
        default='.pdf,.doc,.xls,.zip',
        help='Comma-separated file extensions to exclude (default: .pdf,.doc,.xls,.zip)'
    )
    
    filter_group.add_argument(
        '--conflict-keywords',
        type=str,
        help='Comma-separated conflict-related keywords for monitoring'
    )
    
    # Adaptive crawling options
    adaptive_group = parser.add_argument_group('Adaptive Crawling')
    adaptive_group.add_argument(
        '--disable-adaptive',
        action='store_true',
        help='Disable adaptive crawling strategy'
    )
    
    adaptive_group.add_argument(
        '--strategy-switching-threshold',
        type=float,
        default=0.8,
        help='Success rate threshold for strategy switching (default: 0.8)'
    )
    
    adaptive_group.add_argument(
        '--force-strategy',
        choices=['adaptive', 'http_only', 'browser_only', 'mixed'],
        help='Force specific crawling strategy'
    )
    
    # Stealth options
    stealth_group = parser.add_argument_group('Stealth Options')
    stealth_group.add_argument(
        '--disable-stealth',
        action='store_true',
        help='Disable stealth features'
    )
    
    stealth_group.add_argument(
        '--user-agent',
        type=str,
        help='Custom user agent string'
    )
    
    stealth_group.add_argument(
        '--random-user-agents',
        action='store_true',
        help='Use random user agents'
    )
    
    stealth_group.add_argument(
        '--proxy-list',
        type=str,
        help='Path to proxy list file'
    )
    
    stealth_group.add_argument(
        '--ignore-robots-txt',
        action='store_true',
        help='Ignore robots.txt restrictions'
    )
    
    # Database options
    db_group = parser.add_argument_group('Database Options')
    db_group.add_argument(
        '--enable-database',
        action='store_true',
        help='Enable database storage'
    )
    
    db_group.add_argument(
        '--database-url',
        type=str,
        help='Database connection URL'
    )
    
    db_group.add_argument(
        '--database-table-prefix',
        type=str,
        default='gen_crawler_',
        help='Database table prefix (default: gen_crawler_)'
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    config_group.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to file'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Content Analysis')
    analysis_group.add_argument(
        '--disable-content-analysis',
        action='store_true',
        help='Disable content analysis'
    )
    
    analysis_group.add_argument(
        '--disable-image-extraction',
        action='store_true',
        help='Disable image extraction'
    )
    
    analysis_group.add_argument(
        '--disable-link-analysis',
        action='store_true',
        help='Disable link analysis'
    )
    
    analysis_group.add_argument(
        '--extraction-method',
        choices=['trafilatura', 'newspaper', 'readability', 'beautifulsoup', 'auto'],
        default='auto',
        help='Preferred content extraction method (default: auto)'
    )

def _add_analyze_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the analyze command."""
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to crawl results file or directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for analysis results'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv', 'html'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--conflict-analysis',
        action='store_true',
        help='Perform conflict-related content analysis'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.5,
        help='Quality score threshold for filtering (default: 0.5)'
    )

def _add_config_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the config command."""
    
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create a new configuration file'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        help='Validate configuration file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path for configuration file'
    )
    
    parser.add_argument(
        '--template',
        choices=['basic', 'news', 'research', 'monitoring'],
        default='basic',
        help='Configuration template (default: basic)'
    )

def _add_export_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the export command."""
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to crawl results file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['markdown', 'html', 'pdf', 'csv'],
        default='markdown',
        help='Export format (default: markdown)'
    )
    
    parser.add_argument(
        '--filter-quality',
        type=float,
        help='Filter by minimum quality score'
    )
    
    parser.add_argument(
        '--filter-type',
        type=str,
        help='Filter by content type (e.g., article)'
    )
    
    parser.add_argument(
        '--organize-by',
        choices=['site', 'date', 'type', 'quality'],
        default='site',
        help='Organization structure (default: site)'
    )

def build_config_from_args(args) -> Dict[str, Any]:
    """Build configuration dictionary from CLI arguments."""
    
    config = {}
    
    # Performance settings
    if hasattr(args, 'max_pages'):
        config.setdefault('performance', {})['max_pages_per_site'] = args.max_pages
    if hasattr(args, 'max_concurrent'):
        config.setdefault('performance', {})['max_concurrent'] = args.max_concurrent
    if hasattr(args, 'request_timeout'):
        config.setdefault('performance', {})['request_timeout'] = args.request_timeout
    if hasattr(args, 'max_retries'):
        config.setdefault('performance', {})['max_retries'] = args.max_retries
    if hasattr(args, 'crawl_delay'):
        config.setdefault('performance', {})['crawl_delay'] = args.crawl_delay
    if hasattr(args, 'max_depth'):
        config.setdefault('performance', {})['max_depth'] = args.max_depth
    if hasattr(args, 'memory_limit'):
        config.setdefault('performance', {})['memory_limit_mb'] = args.memory_limit
    
    # Content filter settings
    if hasattr(args, 'min_content_length'):
        config.setdefault('content_filters', {})['min_content_length'] = args.min_content_length
    if hasattr(args, 'max_content_length'):
        config.setdefault('content_filters', {})['max_content_length'] = args.max_content_length
    
    if hasattr(args, 'include_patterns') and args.include_patterns:
        config.setdefault('content_filters', {})['include_patterns'] = [
            p.strip() for p in args.include_patterns.split(',')
        ]
    
    if hasattr(args, 'exclude_patterns') and args.exclude_patterns:
        config.setdefault('content_filters', {})['exclude_patterns'] = [
            p.strip() for p in args.exclude_patterns.split(',')
        ]
    
    if hasattr(args, 'exclude_extensions') and args.exclude_extensions:
        config.setdefault('content_filters', {})['exclude_extensions'] = [
            e.strip() for e in args.exclude_extensions.split(',')
        ]
    
    # Adaptive settings
    if hasattr(args, 'disable_adaptive') and args.disable_adaptive:
        config.setdefault('adaptive', {})['enable_adaptive_crawling'] = False
    
    if hasattr(args, 'strategy_switching_threshold'):
        config.setdefault('adaptive', {})['strategy_switching_threshold'] = args.strategy_switching_threshold
    
    # Stealth settings
    if hasattr(args, 'disable_stealth') and args.disable_stealth:
        config.setdefault('stealth', {})['enable_stealth'] = False
    
    if hasattr(args, 'user_agent') and args.user_agent:
        config.setdefault('stealth', {})['user_agent'] = args.user_agent
    
    if hasattr(args, 'random_user_agents') and args.random_user_agents:
        config.setdefault('stealth', {})['random_user_agents'] = True
    
    if hasattr(args, 'ignore_robots_txt') and args.ignore_robots_txt:
        config.setdefault('stealth', {})['respect_robots_txt'] = False
    
    # Database settings
    if hasattr(args, 'enable_database') and args.enable_database:
        config.setdefault('database', {})['enable_database'] = True
    
    if hasattr(args, 'database_url') and args.database_url:
        config.setdefault('database', {})['connection_string'] = args.database_url
    
    if hasattr(args, 'database_table_prefix'):
        config.setdefault('database', {})['table_prefix'] = args.database_table_prefix
    
    # Content analysis settings
    if hasattr(args, 'disable_content_analysis') and args.disable_content_analysis:
        config['enable_content_analysis'] = False
    
    if hasattr(args, 'disable_image_extraction') and args.disable_image_extraction:
        config['enable_image_extraction'] = False
    
    if hasattr(args, 'disable_link_analysis') and args.disable_link_analysis:
        config['enable_link_analysis'] = False
    
    if hasattr(args, 'save_raw_html') and args.save_raw_html:
        config['save_raw_html'] = True
    
    if hasattr(args, 'compress') and args.compress:
        config['compression_enabled'] = True
    
    return config

async def main_cli():
    """Main CLI entry point."""
    
    parser = create_cli_parser()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = setup_logging(args.verbose, args.quiet, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Import commands here to avoid circular imports
        from .commands import crawl_command, analyze_command, config_command, export_command
        
        # Handle different commands
        if args.command == 'crawl':
            await crawl_command(args)
        elif args.command == 'analyze':
            await analyze_command(args)
        elif args.command == 'config':
            await config_command(args)
        elif args.command == 'export':
            await export_command(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def cli_main():
    """Synchronous wrapper for CLI main function."""
    asyncio.run(main_cli())

if __name__ == "__main__":
    cli_main()