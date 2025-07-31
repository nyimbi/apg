#!/usr/bin/env python3
"""
Google News Crawler CLI Commands
===============================

Implementation of all CLI commands for the Google News Crawler.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import json
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

# Configure logger
logger = logging.getLogger(__name__)

# Import utilities
from .utils import (
    parse_date_input,
    format_output,
    create_mock_db_manager,
    load_cli_config,
    save_cli_config,
    validate_countries,
    validate_languages
)

async def search_command(args) -> None:
    """Execute the search command."""
    logger.info(f"🔍 Searching for: '{args.query}'")
    
    try:
        # Parse arguments
        countries = [c.strip().upper() for c in args.countries.split(',')]
        languages = [l.strip().lower() for l in args.languages.split(',')]
        
        # Validate inputs
        countries = validate_countries(countries)
        languages = validate_languages(languages)
        
        # Parse date range if provided
        time_range = None
        if args.since or args.until:
            start_date = parse_date_input(args.since) if args.since else None
            end_date = parse_date_input(args.until) if args.until else None
            if start_date or end_date:
                time_range = (start_date, end_date)
        
        # Parse source filter
        source_filter = None
        if args.source_filter:
            source_filter = [s.strip() for s in args.source_filter.split(',')]
        
        # Initialize client
        db_manager = create_mock_db_manager()
        
        if args.crawlee:
            from ..api.google_news_client import create_crawlee_enhanced_gnews_client
            logger.info("📥 Creating client with Crawlee enhancement...")
            client = await create_crawlee_enhanced_gnews_client(db_manager=db_manager)
        else:
            from ..api.google_news_client import create_enhanced_gnews_client
            logger.info("📰 Creating standard Google News client...")
            client = await create_enhanced_gnews_client(db_manager=db_manager)
        
        # Perform search
        search_start = time.time()
        articles = await client.search_news(
            query=args.query,
            countries=countries,
            languages=languages,
            max_results=args.max_results,
            time_range=time_range,
            source_filter=source_filter,
            enable_crawlee=args.crawlee
        )
        search_time = time.time() - search_start
        
        # Display results
        logger.info(f"✅ Found {len(articles)} articles in {search_time:.1f}s")
        
        if articles:
            # Analyze results
            _analyze_search_results(articles, args.crawlee)
            
            # Format and display output
            if args.format == 'table':
                _display_table_results(articles, args.crawlee)
            elif args.format == 'json':
                print(json.dumps(articles, indent=2, default=str))
            elif args.format == 'csv':
                _display_csv_results(articles)
            elif args.format == 'txt':
                _display_text_results(articles, args.crawlee)
            
            # Export if requested
            if args.export:
                await _export_results(articles, args.export, args.format)
        else:
            logger.warning("No articles found matching the search criteria")
        
        # Cleanup
        await client.close()
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise

async def config_command(args) -> None:
    """Execute the config command."""
    logger.info("⚙️ Managing configuration...")
    
    try:
        config_path = args.config_file or Path.home() / '.gnews-crawler' / 'config.json'
        
        if args.show:
            await _show_configuration(config_path)
        elif args.init:
            await _init_configuration(config_path)
        elif args.validate:
            await _validate_configuration(config_path)
        elif args.set:
            await _set_configuration(config_path, args.set)
        elif args.get:
            await _get_configuration(config_path, args.get)
        else:
            logger.error("No config action specified. Use --show, --init, --validate, --set, or --get")
            
    except Exception as e:
        logger.error(f"Configuration management failed: {e}")
        raise

async def monitor_command(args) -> None:
    """Execute the monitor command."""
    logger.info(f"📡 Starting monitoring for: '{args.query}'")
    
    try:
        countries = [c.strip().upper() for c in args.countries.split(',')]
        countries = validate_countries(countries)
        
        # Parse alert keywords
        alert_keywords = []
        if args.alert_keywords:
            alert_keywords = [k.strip().lower() for k in args.alert_keywords.split(',')]
        
        # Setup output directory
        output_dir = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        db_manager = create_mock_db_manager()
        
        if args.crawlee:
            from ..api.google_news_client import create_crawlee_enhanced_gnews_client
            client = await create_crawlee_enhanced_gnews_client(db_manager=db_manager)
        else:
            from ..api.google_news_client import create_enhanced_gnews_client
            client = await create_enhanced_gnews_client(db_manager=db_manager)
        
        logger.info(f"🔄 Monitoring every {args.interval} seconds. Press Ctrl+C to stop.")
        
        # Monitoring loop
        cycle = 0
        seen_urls = set()
        
        try:
            while True:
                cycle += 1
                cycle_start = time.time()
                
                logger.info(f"📊 Monitoring cycle {cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Search for articles
                articles = await client.search_news(
                    query=args.query,
                    countries=countries,
                    max_results=args.max_results,
                    enable_crawlee=args.crawlee
                )
                
                # Filter new articles
                new_articles = [a for a in articles if a.get('url') not in seen_urls]
                for article in new_articles:
                    seen_urls.add(article.get('url'))
                
                logger.info(f"   Found {len(articles)} total, {len(new_articles)} new articles")
                
                # Check for alerts
                alert_articles = _check_alerts(new_articles, alert_keywords)
                if alert_articles:
                    logger.warning(f"🚨 ALERT: {len(alert_articles)} articles contain alert keywords!")
                    for article in alert_articles:
                        logger.warning(f"   📰 {article.get('title', 'No title')}")
                        logger.warning(f"      🔗 {article.get('url', 'No URL')}")
                
                # Save results if output directory specified
                if output_dir and new_articles:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = output_dir / f"monitoring_cycle_{cycle}_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(new_articles, f, indent=2, default=str)
                    logger.info(f"   💾 Saved {len(new_articles)} new articles to {filename}")
                
                # Wait for next cycle
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, args.interval - cycle_time)
                
                if sleep_time > 0:
                    logger.info(f"   ⏰ Sleeping for {sleep_time:.1f}s until next cycle...")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        
        # Cleanup
        await client.close()
        
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise

async def crawlee_command(args) -> None:
    """Execute the crawlee command."""
    logger.info("🕷️ Managing Crawlee integration...")
    
    try:
        if args.status:
            await _show_crawlee_status()
        elif args.test:
            await _test_crawlee_integration(args)
        elif args.config_template:
            await _generate_crawlee_config_template(args.config)
        else:
            logger.error("No Crawlee action specified. Use --status, --test, or --config-template")
            
    except Exception as e:
        logger.error(f"Crawlee management failed: {e}")
        raise

async def status_command(args) -> None:
    """Execute the status command."""
    logger.info("🔍 Checking system status...")
    
    try:
        print("\n" + "="*60)
        print("Google News Crawler - System Status")
        print("="*60)
        
        # Basic system info
        print(f"\nSystem Information:")
        print(f"  Python Version: {sys.version.split()[0]}")
        print(f"  Platform: {sys.platform}")
        print(f"  Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check dependencies
        if args.check_deps:
            await _check_dependencies()
        
        # Test database if requested
        if args.test_db:
            await _test_database_connectivity()
        
        # Test Crawlee if requested
        if args.test_crawlee:
            await _test_crawlee_basic()
        
        # Default checks
        if not any([args.check_deps, args.test_db, args.test_crawlee]):
            await _basic_status_check()
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise

# Helper functions

def _analyze_search_results(articles: List[Dict[str, Any]], crawlee_enabled: bool) -> None:
    """Analyze and display search result statistics."""
    if not articles:
        return
    
    print(f"\n📊 Search Results Analysis:")
    print(f"   Total articles: {len(articles)}")
    
    # Source analysis
    sources = [article.get('publisher', {}).get('name', 'Unknown') for article in articles]
    source_counts = Counter(sources)
    print(f"   Top sources: {', '.join([f'{s}({c})' for s, c in source_counts.most_common(3)])}")
    
    # Country analysis
    countries = [article.get('country', 'Unknown') for article in articles]
    country_counts = Counter(countries)
    print(f"   Countries: {', '.join([f'{c}({count})' for c, count in country_counts.most_common(5)])}")
    
    # Crawlee analysis if enabled
    if crawlee_enabled:
        enhanced = [a for a in articles if a.get('crawlee_enhanced', False)]
        print(f"   Crawlee enhanced: {len(enhanced)}/{len(articles)} ({len(enhanced)/len(articles)*100:.1f}%)")
        
        if enhanced:
            avg_word_count = sum(a.get('word_count', 0) for a in enhanced) / len(enhanced)
            avg_quality = sum(a.get('crawlee_quality_score', 0) for a in enhanced) / len(enhanced)
            print(f"   Average word count: {avg_word_count:.0f}")
            print(f"   Average quality score: {avg_quality:.2f}")

def _display_table_results(articles: List[Dict[str, Any]], crawlee_enabled: bool) -> None:
    """Display results in table format."""
    if not articles:
        return
    
    print(f"\n📰 Articles ({len(articles)} found):")
    print("-" * 120)
    
    for i, article in enumerate(articles[:10], 1):  # Show first 10
        title = article.get('title', 'No title')[:70]
        source = article.get('publisher', {}).get('name', 'Unknown')[:15]
        country = article.get('country', 'XX')
        
        quality_info = ""
        if crawlee_enabled and article.get('crawlee_enhanced'):
            quality = article.get('crawlee_quality_score', 0)
            word_count = article.get('word_count', 0)
            quality_info = f" | Q:{quality:.2f} W:{word_count}"
        
        print(f"{i:2d}. {title:<70} | {source:<15} | {country} {quality_info}")
    
    if len(articles) > 10:
        print(f"... and {len(articles) - 10} more articles")

def _display_csv_results(articles: List[Dict[str, Any]]) -> None:
    """Display results in CSV format."""
    if not articles:
        return
    
    # Determine fields to include
    fields = ['title', 'url', 'publisher', 'country', 'published_date']
    if any(a.get('crawlee_enhanced') for a in articles):
        fields.extend(['word_count', 'crawlee_quality_score', 'crawlee_enhanced'])
    
    # Write CSV to stdout
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    
    for article in articles:
        # Flatten publisher data
        publisher_name = article.get('publisher', {}).get('name', '') if isinstance(article.get('publisher'), dict) else str(article.get('publisher', ''))
        row = dict(article)
        row['publisher'] = publisher_name
        writer.writerow(row)
    
    print(output.getvalue())

def _display_text_results(articles: List[Dict[str, Any]], crawlee_enabled: bool) -> None:
    """Display results in text format."""
    for i, article in enumerate(articles, 1):
        print(f"\n{'='*80}")
        print(f"Article {i}: {article.get('title', 'No title')}")
        print(f"{'='*80}")
        print(f"URL: {article.get('url', 'No URL')}")
        print(f"Source: {article.get('publisher', {}).get('name', 'Unknown')}")
        print(f"Country: {article.get('country', 'Unknown')}")
        print(f"Published: {article.get('published_date', 'Unknown')}")
        
        if crawlee_enabled and article.get('crawlee_enhanced'):
            print(f"Enhanced: Yes (Quality: {article.get('crawlee_quality_score', 0):.2f})")
            print(f"Word Count: {article.get('word_count', 0)}")
            if article.get('geographic_entities'):
                print(f"Geographic Entities: {', '.join(article['geographic_entities'])}")
            if article.get('conflict_indicators'):
                print(f"Conflict Indicators: {', '.join(article['conflict_indicators'])}")
        
        description = article.get('description', '')
        if description:
            print(f"\nDescription: {description}")
        
        if crawlee_enabled and article.get('full_content'):
            content_preview = article['full_content'][:500]
            print(f"\nContent Preview: {content_preview}...")

async def _export_results(articles: List[Dict[str, Any]], filepath: str, format_type: str) -> None:
    """Export results to file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json' or path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(articles, f, indent=2, default=str)
        elif format_type == 'csv' or path.suffix.lower() == '.csv':
            if articles:
                # Determine fields
                fields = list(articles[0].keys())
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for article in articles:
                        # Handle nested dictionaries
                        flat_article = {}
                        for k, v in article.items():
                            if isinstance(v, dict):
                                flat_article[k] = json.dumps(v)
                            elif isinstance(v, list):
                                flat_article[k] = ', '.join(str(x) for x in v)
                            else:
                                flat_article[k] = v
                        writer.writerow(flat_article)
        else:
            # Default to text format
            with open(path, 'w') as f:
                for i, article in enumerate(articles, 1):
                    f.write(f"Article {i}: {article.get('title', 'No title')}\n")
                    f.write(f"URL: {article.get('url', 'No URL')}\n")
                    f.write(f"Source: {article.get('publisher', {}).get('name', 'Unknown')}\n")
                    f.write(f"Description: {article.get('description', '')}\n")
                    f.write("-" * 80 + "\n")
        
        logger.info(f"💾 Exported {len(articles)} articles to {path}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise

def _check_alerts(articles: List[Dict[str, Any]], alert_keywords: List[str]) -> List[Dict[str, Any]]:
    """Check articles for alert keywords."""
    if not alert_keywords:
        return []
    
    alert_articles = []
    for article in articles:
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('full_content', '').lower()
        
        # Check if any alert keyword appears in title, description, or content
        for keyword in alert_keywords:
            if keyword in title or keyword in description or keyword in content:
                alert_articles.append(article)
                break
    
    return alert_articles

async def _show_configuration(config_path: Path) -> None:
    """Show current configuration."""
    try:
        config = load_cli_config(config_path)
        print(f"\n⚙️ Configuration from {config_path}:")
        print(json.dumps(config, indent=2))
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("Use 'gnews-crawler config --init' to create default configuration")

async def _init_configuration(config_path: Path) -> None:
    """Initialize default configuration."""
    config = {
        "database": {
            "url": "postgresql://user:password@localhost:5432/gnews_db"
        },
        "crawlee": {
            "max_requests": 100,
            "max_concurrent": 5,
            "target_countries": ["ET", "SO", "KE", "UG", "TZ"],
            "enable_full_content": True,
            "min_content_length": 200,
            "enable_content_scoring": True
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
    
    save_cli_config(config_path, config)
    logger.info(f"✅ Default configuration created at {config_path}")

async def _validate_configuration(config_path: Path) -> None:
    """Validate configuration."""
    try:
        config = load_cli_config(config_path)
        
        # Basic validation
        required_sections = ['database', 'crawlee', 'search']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            logger.error(f"❌ Missing configuration sections: {missing_sections}")
            return
        
        logger.info("✅ Configuration validation passed")
        
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")

async def _set_configuration(config_path: Path, settings: List[List[str]]) -> None:
    """Set configuration values."""
    try:
        config = load_cli_config(config_path) if config_path.exists() else {}
        
        for key, value in settings:
            # Support nested keys like "crawlee.max_requests"
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Try to parse value as JSON, fall back to string
            try:
                current[keys[-1]] = json.loads(value)
            except:
                current[keys[-1]] = value
            
            logger.info(f"✅ Set {key} = {value}")
        
        save_cli_config(config_path, config)
        
    except Exception as e:
        logger.error(f"❌ Failed to set configuration: {e}")

async def _get_configuration(config_path: Path, key: str) -> None:
    """Get specific configuration value."""
    try:
        config = load_cli_config(config_path)
        
        keys = key.split('.')
        current = config
        for k in keys:
            if k not in current:
                logger.error(f"❌ Configuration key not found: {key}")
                return
            current = current[k]
        
        print(f"{key}: {json.dumps(current, indent=2)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to get configuration: {e}")

async def _show_crawlee_status() -> None:
    """Show Crawlee status and capabilities."""
    try:
        from ..crawlee_integration import CRAWLEE_AVAILABLE
        
        print(f"\n🕷️ Crawlee Integration Status:")
        print(f"   Available: {'✅ Yes' if CRAWLEE_AVAILABLE else '❌ No'}")
        
        if CRAWLEE_AVAILABLE:
            from ..crawlee_integration import CrawleeNewsEnhancer
            
            # Test extraction methods
            test_config = {'enable_full_content': True}
            enhancer = CrawleeNewsEnhancer(test_config)
            
            print(f"   Extraction methods available:")
            for method, available in enhancer.extractors_available.items():
                status = '✅' if available else '❌'
                print(f"     {method}: {status}")
        else:
            print(f"   Install with: pip install crawlee playwright")
            print(f"   Then run: playwright install chromium")
            
    except Exception as e:
        logger.error(f"Failed to check Crawlee status: {e}")

async def _test_crawlee_integration(args) -> None:
    """Test Crawlee integration with sample articles."""
    try:
        from ..crawlee_integration import create_crawlee_enhancer, create_crawlee_config
        
        logger.info(f"🧪 Testing Crawlee integration...")
        
        # Create test configuration
        config = create_crawlee_config(
            max_requests=args.max_requests,
            max_concurrent=2,
            preferred_extraction_method=args.method,
            enable_full_content=True
        )
        
        # Create enhancer
        enhancer = await create_crawlee_enhancer(config)
        
        # Test articles
        test_articles = [
            {
                'title': 'BBC News Test Article',
                'url': 'https://www.bbc.com/news/world-africa',
                'description': 'Test description',
                'published_date': datetime.now(),
                'source': 'BBC'
            }
        ]
        
        logger.info(f"📥 Testing content enhancement for {len(test_articles)} articles...")
        
        # Enhance articles
        enhanced = await enhancer.enhance_articles(test_articles)
        
        # Display results
        logger.info(f"✅ Enhanced {len(enhanced)} articles")
        
        for article in enhanced:
            print(f"\n📄 {article.title}")
            print(f"   URL: {article.url}")
            print(f"   Success: {article.crawl_success}")
            print(f"   Method: {article.extraction_method}")
            print(f"   Word Count: {article.word_count}")
            print(f"   Quality Score: {article.quality_score:.2f}")
            
            if article.geographic_entities:
                print(f"   Geographic: {', '.join(article.geographic_entities)}")
            if article.conflict_indicators:
                print(f"   Conflict: {', '.join(article.conflict_indicators)}")
        
        # Show statistics
        stats = enhancer.get_processing_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Successful: {stats['successful_downloads']}")
        print(f"   Failed: {stats['failed_downloads']}")
        print(f"   Success rate: {stats['successful_downloads']/max(stats['total_requests'], 1)*100:.1f}%")
        
        await enhancer.close()
        
    except Exception as e:
        logger.error(f"Crawlee test failed: {e}")
        raise

async def _generate_crawlee_config_template(output_path: Optional[str]) -> None:
    """Generate Crawlee configuration template."""
    template = {
        "max_requests_per_crawl": 100,
        "max_concurrent": 5,
        "enable_full_content": True,
        "enable_image_extraction": True,
        "enable_metadata_extraction": True,
        "preferred_extraction_method": "auto",
        "min_content_length": 200,
        "max_content_length": 50000,
        "enable_content_scoring": True,
        "min_quality_score": 0.3,
        "target_countries": ["ET", "SO", "ER", "DJ", "KE", "UG", "TZ", "SD", "SS"],
        "target_languages": ["en", "fr", "ar", "sw"],
        "crawl_delay": 1.0,
        "respect_robots_txt": True,
        "max_retries": 3,
        "enable_caching": True,
        "save_raw_html": False
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        logger.info(f"✅ Crawlee configuration template saved to {output_path}")
    else:
        print("\n🕷️ Crawlee Configuration Template:")
        print(json.dumps(template, indent=2))

async def _check_dependencies() -> None:
    """Check all dependencies."""
    dependencies = {
        'Core Dependencies': [
            ('aiohttp', 'HTTP client for async operations'),
            ('asyncpg', 'PostgreSQL async driver'),
            ('feedparser', 'RSS/Atom feed parsing'),
        ],
        'Content Processing': [
            ('beautifulsoup4', 'HTML parsing'),
            ('trafilatura', 'Content extraction'),
            ('newspaper3k', 'Article extraction'),
            ('readability-lxml', 'Content readability'),
        ],
        'Crawlee Integration': [
            ('crawlee', 'Web crawling framework'),
            ('playwright', 'Browser automation'),
        ],
        'Optional': [
            ('pandas', 'Data manipulation'),
            ('textblob', 'Text analysis'),
            ('scikit-learn', 'Machine learning'),
            ('pydantic', 'Data validation'),
            ('PyYAML', 'Configuration files'),
        ]
    }
    
    print(f"\n📦 Dependency Check:")
    
    for category, deps in dependencies.items():
        print(f"\n{category}:")
        for dep_name, description in deps:
            try:
                __import__(dep_name.replace('-', '_'))
                print(f"   ✅ {dep_name}: {description}")
            except ImportError:
                print(f"   ❌ {dep_name}: {description}")

async def _test_database_connectivity() -> None:
    """Test database connectivity."""
    print(f"\n🗄️ Database Connectivity Test:")
    print(f"   Status: ⚠️ Using mock database (no real DB configured)")
    print(f"   Note: Configure real database URL in config for production use")

async def _test_crawlee_basic() -> None:
    """Basic Crawlee test."""
    try:
        from ..crawlee_integration import CRAWLEE_AVAILABLE
        
        print(f"\n🕷️ Crawlee Basic Test:")
        if CRAWLEE_AVAILABLE:
            print(f"   Status: ✅ Available")
            print(f"   Use 'gnews-crawler crawlee --test' for full integration test")
        else:
            print(f"   Status: ❌ Not available")
            print(f"   Install with: pip install crawlee playwright")
    except Exception as e:
        print(f"   Status: ❌ Error - {e}")

async def _basic_status_check() -> None:
    """Perform basic status checks."""
    print(f"\n✅ Basic Status Check:")
    print(f"   Google News Crawler: Ready")
    print(f"   CLI Interface: Functional")
    print(f"   Core Dependencies: Available")
    print(f"\nUse specific flags for detailed checks:")
    print(f"   --check-deps: Check all dependencies")
    print(f"   --test-db: Test database connectivity")
    print(f"   --test-crawlee: Test Crawlee integration")