#!/usr/bin/env python3
"""
Google News Crawler CLI - Main Entry Point
==========================================

Main command-line interface for Google News Crawler with comprehensive
subcommands for searching, configuration, monitoring, and Crawlee integration.

Usage:
    gnews-crawler search "Ethiopia conflict" --countries ET,SO --crawlee
    gnews-crawler config --show
    gnews-crawler monitor --interval 300 --query "Horn of Africa crisis"
    gnews-crawler crawlee --test --config crawlee_config.json

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import argparse
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup comprehensive argument parser for Google News Crawler CLI."""
    
    parser = argparse.ArgumentParser(
        prog='gnews-crawler',
        description='Google News Crawler - Enterprise-grade news intelligence with Crawlee integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic news search
  gnews-crawler search "Ethiopia conflict" --max-results 50

  # Enhanced search with Crawlee content downloading
  gnews-crawler search "Somalia crisis" --crawlee --countries SO,ET,KE

  # Monitor news continuously
  gnews-crawler monitor --interval 300 --query "Horn of Africa"

  # Test Crawlee integration
  gnews-crawler crawlee --test --max-requests 5

  # Show configuration
  gnews-crawler config --show

  # Export search results
  gnews-crawler search "Sudan violence" --export results.json

For more information, visit: https://github.com/datacraft/lindela
        """
    )

    # Global options
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Google News Crawler CLI v1.0.0'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--db-url',
        type=str,
        help='Database connection URL (overrides config)'
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )

    # Search command
    search_parser = subparsers.add_parser(
        'search',
        help='Search for news articles',
        description='Search Google News with optional Crawlee enhancement'
    )
    
    search_parser.add_argument(
        'query',
        type=str,
        help='Search query (supports boolean operators)'
    )
    
    search_parser.add_argument(
        '--countries', '-c',
        type=str,
        default='ET,SO,KE,SD,SS,UG,TZ',
        help='Comma-separated country codes (default: Horn of Africa)'
    )
    
    search_parser.add_argument(
        '--languages', '-l',
        type=str,
        default='en,fr,ar',
        help='Comma-separated language codes (default: en,fr,ar)'
    )
    
    search_parser.add_argument(
        '--max-results', '-n',
        type=int,
        default=100,
        help='Maximum number of results (default: 100)'
    )
    
    search_parser.add_argument(
        '--crawlee',
        action='store_true',
        help='Enable Crawlee content enhancement'
    )
    
    search_parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export results to file (JSON, CSV, or TXT)'
    )
    
    search_parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv', 'txt', 'table'],
        default='table',
        help='Output format (default: table)'
    )
    
    search_parser.add_argument(
        '--source-filter',
        type=str,
        help='Comma-separated list of domains to include/exclude'
    )
    
    search_parser.add_argument(
        '--since',
        type=str,
        help='Search articles since date (YYYY-MM-DD or relative like "7d", "24h")'
    )
    
    search_parser.add_argument(
        '--until',
        type=str,
        help='Search articles until date (YYYY-MM-DD)'
    )

    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Manage configuration',
        description='View and modify Google News Crawler configuration'
    )
    
    config_group = config_parser.add_mutually_exclusive_group()
    config_group.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration'
    )
    
    config_group.add_argument(
        '--init',
        action='store_true',
        help='Initialize default configuration'
    )
    
    config_group.add_argument(
        '--validate',
        action='store_true',
        help='Validate current configuration'
    )
    
    config_parser.add_argument(
        '--set',
        action='append',
        nargs=2,
        metavar=('KEY', 'VALUE'),
        help='Set configuration value (can be used multiple times)'
    )
    
    config_parser.add_argument(
        '--get',
        type=str,
        help='Get specific configuration value'
    )

    # Monitor command
    monitor_parser = subparsers.add_parser(
        'monitor',
        help='Monitor news continuously',
        description='Continuously monitor news with configurable intervals'
    )
    
    monitor_parser.add_argument(
        '--query', '-q',
        type=str,
        required=True,
        help='Search query to monitor'
    )
    
    monitor_parser.add_argument(
        '--interval', '-i',
        type=int,
        default=300,
        help='Monitoring interval in seconds (default: 300)'
    )
    
    monitor_parser.add_argument(
        '--countries',
        type=str,
        default='ET,SO,KE',
        help='Countries to monitor (default: ET,SO,KE)'
    )
    
    monitor_parser.add_argument(
        '--crawlee',
        action='store_true',
        help='Enable Crawlee enhancement for monitoring'
    )
    
    monitor_parser.add_argument(
        '--max-results',
        type=int,
        default=50,
        help='Maximum results per monitoring cycle (default: 50)'
    )
    
    monitor_parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save monitoring results'
    )
    
    monitor_parser.add_argument(
        '--alert-keywords',
        type=str,
        help='Comma-separated keywords that trigger alerts'
    )

    # Crawlee command
    crawlee_parser = subparsers.add_parser(
        'crawlee',
        help='Crawlee integration management',
        description='Test and configure Crawlee integration'
    )
    
    crawlee_group = crawlee_parser.add_mutually_exclusive_group()
    crawlee_group.add_argument(
        '--test',
        action='store_true',
        help='Test Crawlee integration'
    )
    
    crawlee_group.add_argument(
        '--status',
        action='store_true',
        help='Show Crawlee status and capabilities'
    )
    
    crawlee_group.add_argument(
        '--config-template',
        action='store_true',
        help='Generate Crawlee configuration template'
    )
    
    crawlee_parser.add_argument(
        '--config',
        type=str,
        help='Path to Crawlee configuration file'
    )
    
    crawlee_parser.add_argument(
        '--max-requests',
        type=int,
        default=10,
        help='Maximum requests for testing (default: 10)'
    )
    
    crawlee_parser.add_argument(
        '--method',
        choices=['trafilatura', 'newspaper', 'readability', 'beautifulsoup', 'auto'],
        default='auto',
        help='Content extraction method to test (default: auto)'
    )

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show system status',
        description='Display system status, dependencies, and health checks'
    )
    
    status_parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check all dependencies'
    )
    
    status_parser.add_argument(
        '--test-db',
        action='store_true',
        help='Test database connectivity'
    )
    
    status_parser.add_argument(
        '--test-crawlee',
        action='store_true',
        help='Test Crawlee integration'
    )

    return parser

async def main_cli():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Handle no command
    if not args.command:
        parser.print_help()
        return

    try:
        # Import commands (delay import to improve startup time)
        from .commands import (
            search_command,
            config_command,
            monitor_command,
            crawlee_command,
            status_command
        )

        # Route to appropriate command
        if args.command == 'search':
            await search_command(args)
        elif args.command == 'config':
            await config_command(args)
        elif args.command == 'monitor':
            await monitor_command(args)
        elif args.command == 'crawlee':
            await crawlee_command(args)
        elif args.command == 'status':
            await status_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        if args.verbose:
            logger.exception(f"Command failed: {e}")
        else:
            logger.error(f"Command failed: {e}")
        sys.exit(1)

def cli_entry_point():
    """Entry point for console script."""
    asyncio.run(main_cli())

if __name__ == "__main__":
    cli_entry_point()