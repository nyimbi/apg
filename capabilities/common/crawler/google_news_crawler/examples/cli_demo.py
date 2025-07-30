#!/usr/bin/env python3
"""
Google News Crawler CLI Demo
============================

Demonstration script showing CLI functionality without installing the console script.

This script simulates CLI commands by calling the CLI functions directly,
useful for testing and demonstration purposes.

Usage:
    python cli_demo.py

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse

# Add the CLI path for imports
cli_path = Path(__file__).parent.parent / 'cli'
sys.path.insert(0, str(cli_path))

async def demo_search():
    """Demonstrate search functionality."""
    print("🔍 Demo: Search Command")
    print("-" * 40)
    
    try:
        from cli.commands import search_command
        from cli.utils import create_mock_db_manager
        
        # Create mock arguments for search
        class MockSearchArgs:
            query = "Ethiopia conflict"
            countries = "ET,SO,KE"
            languages = "en,fr"
            max_results = 10
            crawlee = False
            export = None
            format = "table"
            source_filter = None
            since = None
            until = None
        
        args = MockSearchArgs()
        print(f"Query: '{args.query}'")
        print(f"Countries: {args.countries}")
        print(f"Max Results: {args.max_results}")
        print(f"Crawlee: {args.crawlee}")
        
        # Note: In a real demo, this would call the actual search_command
        # For now, we just show the argument processing
        print("✅ Search arguments processed successfully")
        print("📝 Note: Full search requires database connection\n")
        
    except Exception as e:
        print(f"❌ Search demo failed: {e}\n")

async def demo_config():
    """Demonstrate configuration functionality."""
    print("⚙️ Demo: Config Command")
    print("-" * 40)
    
    try:
        from cli.utils import save_cli_config, load_cli_config
        import tempfile
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        # Demo configuration
        demo_config = {
            "database": {
                "url": "postgresql://user:password@localhost:5432/gnews_db"
            },
            "crawlee": {
                "max_requests": 50,
                "max_concurrent": 3,
                "enable_full_content": True
            },
            "search": {
                "default_countries": ["ET", "SO", "KE"],
                "default_languages": ["en", "fr", "ar"]
            }
        }
        
        # Save and load config
        save_cli_config(config_path, demo_config)
        loaded_config = load_cli_config(config_path)
        
        print("Configuration saved and loaded:")
        print(f"  Max Requests: {loaded_config['crawlee']['max_requests']}")
        print(f"  Default Countries: {loaded_config['search']['default_countries']}")
        print("✅ Configuration demo completed")
        
        # Cleanup
        config_path.unlink()
        print()
        
    except Exception as e:
        print(f"❌ Config demo failed: {e}\n")

async def demo_crawlee_status():
    """Demonstrate Crawlee status check."""
    print("🕷️ Demo: Crawlee Status")
    print("-" * 40)
    
    # Check extraction methods availability
    extractors = {
        'trafilatura': False,
        'newspaper3k': False,
        'readability': False,
        'beautifulsoup4': False
    }
    
    try:
        import trafilatura
        extractors['trafilatura'] = True
    except ImportError:
        pass
    
    try:
        from newspaper import Article
        extractors['newspaper3k'] = True
    except ImportError:
        pass
    
    try:
        from readability import Document
        extractors['readability'] = True
    except ImportError:
        pass
    
    try:
        from bs4 import BeautifulSoup
        extractors['beautifulsoup4'] = True
    except ImportError:
        pass
    
    print("Content extraction methods:")
    for method, available in extractors.items():
        status = "✅" if available else "❌"
        print(f"  {method}: {status}")
    
    available_count = sum(extractors.values())
    print(f"\nAvailable: {available_count}/4 extraction methods")
    
    if available_count > 0:
        print("✅ Content extraction available")
    else:
        print("⚠️ No extraction methods available")
        print("Install with: pip install trafilatura newspaper3k readability-lxml beautifulsoup4")
    
    print()

async def demo_status_check():
    """Demonstrate system status check."""
    print("🔍 Demo: System Status")
    print("-" * 40)
    
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    # Check core dependencies
    core_deps = {
        'aiohttp': False,
        'asyncpg': False,
        'feedparser': False,
        'beautifulsoup4': False
    }
    
    for dep in core_deps:
        try:
            __import__(dep.replace('-', '_'))
            core_deps[dep] = True
        except ImportError:
            pass
    
    print("\nCore dependencies:")
    for dep, available in core_deps.items():
        status = "✅" if available else "❌"
        print(f"  {dep}: {status}")
    
    available_core = sum(core_deps.values())
    print(f"\nCore dependencies: {available_core}/{len(core_deps)} available")
    print()

async def demo_monitor_setup():
    """Demonstrate monitoring setup."""
    print("📡 Demo: Monitor Setup")
    print("-" * 40)
    
    try:
        from cli.utils import parse_date_input, validate_countries
        
        # Demo monitoring parameters
        query = "Horn of Africa crisis"
        interval = 300  # 5 minutes
        countries = validate_countries(['ET', 'SO', 'KE'])
        alert_keywords = ['violence', 'attack', 'displacement']
        
        print(f"Query: '{query}'")
        print(f"Interval: {interval} seconds ({interval//60} minutes)")
        print(f"Countries: {countries}")
        print(f"Alert Keywords: {alert_keywords}")
        
        # Test date parsing
        since_date = parse_date_input("7d")
        print(f"Since Date: {since_date.strftime('%Y-%m-%d %H:%M:%S') if since_date else 'None'}")
        
        print("✅ Monitor setup parameters validated")
        print("📝 Note: Actual monitoring requires database connection")
        print()
        
    except Exception as e:
        print(f"❌ Monitor demo failed: {e}\n")

async def demo_cli_help():
    """Demonstrate CLI help information."""
    print("📚 Demo: CLI Help Information")
    print("-" * 40)
    
    print("Available Commands:")
    print("  search    - Search for news articles")
    print("  monitor   - Continuous news monitoring")
    print("  config    - Configuration management")
    print("  crawlee   - Crawlee integration tools")
    print("  status    - System status and health")
    
    print("\nExample Usage:")
    print("  gnews-crawler search 'Ethiopia conflict' --countries ET,SO")
    print("  gnews-crawler monitor --query 'Horn of Africa' --interval 300")
    print("  gnews-crawler config --show")
    print("  gnews-crawler crawlee --status")
    print("  gnews-crawler status --check-deps")
    
    print("\nFor detailed help:")
    print("  gnews-crawler --help")
    print("  gnews-crawler search --help")
    print()

async def main():
    """Run all CLI demos."""
    print("🚀 Google News Crawler CLI Demo")
    print("=" * 50)
    print("This demo shows CLI functionality without requiring full installation.\n")
    
    demos = [
        demo_search,
        demo_config,
        demo_crawlee_status,
        demo_status_check,
        demo_monitor_setup,
        demo_cli_help
    ]
    
    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"❌ Demo failed: {e}\n")
    
    print("=" * 50)
    print("🎉 CLI Demo Complete!")
    print("\nTo install and use the actual CLI:")
    print("1. cd to the google_news_crawler directory")
    print("2. Run: pip install -e .")
    print("3. Use: gnews-crawler --help")
    print("\nFor Crawlee integration:")
    print("1. Run: pip install 'crawlee>=0.1.0' 'playwright>=1.30.0'")
    print("2. Run: playwright install chromium")
    print("3. Test: gnews-crawler crawlee --status")

if __name__ == "__main__":
    asyncio.run(main())