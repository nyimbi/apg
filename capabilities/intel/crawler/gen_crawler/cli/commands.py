"""
Gen Crawler CLI Commands
========================

Implementation of CLI commands for the generation crawler.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse

from .utils import validate_urls, format_results
from .exporters import MarkdownExporter, JSONExporter, CSVExporter, HTMLExporter
# Removed circular import - build_config_from_args is defined locally
try:
    from ..core import GenCrawler, create_gen_crawler
    from ..config import create_gen_config, GenCrawlerSettings
except ImportError:
    # Handle relative import issues in testing
    import sys
    from pathlib import Path
    
    # Add parent directories to path
    current_dir = Path(__file__).parent
    package_dir = current_dir.parent
    sys.path.insert(0, str(package_dir))
    
    from core import GenCrawler, create_gen_crawler
    from config import create_gen_config, GenCrawlerSettings

logger = logging.getLogger(__name__)

def _build_config_from_args(args) -> Dict[str, Any]:
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

async def crawl_command(args: argparse.Namespace):
    """Execute the crawl command."""
    
    logger.info(f"🚀 Starting crawl command for {len(args.urls)} URL(s)")
    
    # Validate URLs
    valid_urls, invalid_urls = validate_urls(args.urls)
    
    if invalid_urls:
        logger.warning(f"Invalid URLs detected: {invalid_urls}")
        if not valid_urls:
            logger.error("No valid URLs to crawl")
            sys.exit(1)
    
    # Load or create configuration
    if hasattr(args, 'config') and args.config:
        logger.info(f"Loading configuration from: {args.config}")
        try:
            config_manager = create_gen_config(config_file=args.config)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    else:
        # Build config from CLI arguments
        config_dict = _build_config_from_args(args)
        config_manager = create_gen_config()
        
        # Update with CLI arguments
        if config_dict:
            # Update settings with CLI arguments
            for section, values in config_dict.items():
                if hasattr(config_manager.settings, section):
                    section_obj = getattr(config_manager.settings, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                else:
                    # Top-level settings
                    setattr(config_manager.settings, section, values)
    
    # Save configuration if requested
    if hasattr(args, 'save_config') and args.save_config:
        try:
            config_manager.save_to_file(args.save_config)
            logger.info(f"Configuration saved to: {args.save_config}")
        except Exception as e:
            logger.warning(f"Could not save configuration: {e}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create crawler
    crawler_config = config_manager.get_crawler_config()
    crawler = GenCrawler(crawler_config)
    
    try:
        await crawler.initialize()
        
        # Process conflict keywords if provided
        conflict_keywords = []
        if hasattr(args, 'conflict_keywords') and args.conflict_keywords:
            conflict_keywords = [k.strip() for k in args.conflict_keywords.split(',')]
        
        # Crawl each site
        all_results = []
        
        for i, url in enumerate(valid_urls, 1):
            logger.info(f"📡 Crawling site {i}/{len(valid_urls)}: {url}")
            
            try:
                result = await crawler.crawl_site(url)
                
                # Add conflict analysis if keywords provided
                if conflict_keywords and result.pages:
                    result = _add_conflict_analysis(result, conflict_keywords)
                
                all_results.append(result)
                
                logger.info(f"✅ Completed {url}: {result.total_pages} pages, "
                           f"{result.success_rate:.1f}% success rate")
                
            except Exception as e:
                logger.error(f"❌ Failed to crawl {url}: {e}")
                continue
        
        # Export results
        await _export_crawl_results(all_results, args, output_dir)
        
        # Display summary
        _display_crawl_summary(all_results, conflict_keywords)
        
    finally:
        await crawler.cleanup()

def _add_conflict_analysis(result, conflict_keywords: List[str]):
    """Add conflict analysis to crawl results."""
    
    conflict_count = 0
    
    for page in result.pages:
        if not page.content:
            continue
            
        content_lower = page.content.lower()
        title_lower = page.title.lower() if page.title else ""
        
        # Check for conflict keywords
        conflict_matches = sum(
            1 for keyword in conflict_keywords 
            if keyword.lower() in content_lower or keyword.lower() in title_lower
        )
        
        if conflict_matches > 0:
            conflict_count += 1
            page.metadata['conflict_related'] = True
            page.metadata['conflict_matches'] = conflict_matches
            page.metadata['conflict_keywords_found'] = [
                kw for kw in conflict_keywords 
                if kw.lower() in content_lower or kw.lower() in title_lower
            ]
        else:
            page.metadata['conflict_related'] = False
    
    # Add conflict statistics to site result
    result.site_metadata['conflict_analysis'] = {
        'total_conflict_pages': conflict_count,
        'conflict_percentage': (conflict_count / len(result.pages) * 100) if result.pages else 0,
        'keywords_used': conflict_keywords
    }
    
    return result

async def _export_crawl_results(results: List[Any], args: argparse.Namespace, output_dir: Path):
    """Export crawl results in the specified format."""
    
    logger.info(f"📄 Exporting results in {args.format} format")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if args.format == 'markdown':
            exporter = MarkdownExporter(
                output_dir=output_dir,
                save_images=not getattr(args, 'disable_image_extraction', False),
                organize_by_site=True
            )
            await exporter.export_results(results)
            
        elif args.format == 'json':
            exporter = JSONExporter(
                output_dir=output_dir,
                pretty_print=True,
                compress=getattr(args, 'compress', False)
            )
            await exporter.export_results(results)
            
        elif args.format == 'csv':
            exporter = CSVExporter(output_dir=output_dir)
            await exporter.export_results(results)
            
        elif args.format == 'html':
            exporter = HTMLExporter(
                output_dir=output_dir,
                include_images=not getattr(args, 'disable_image_extraction', False)
            )
            await exporter.export_results(results)
        
        # Save raw HTML if requested
        if getattr(args, 'save_raw_html', False):
            raw_dir = output_dir / 'raw_html'
            raw_dir.mkdir(exist_ok=True)
            await _save_raw_html(results, raw_dir)
        
        logger.info(f"✅ Results exported to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise

async def _save_raw_html(results: List[Any], raw_dir: Path):
    """Save raw HTML files."""
    
    for result in results:
        site_name = result.base_url.replace('https://', '').replace('http://', '').replace('/', '_')
        site_dir = raw_dir / site_name
        site_dir.mkdir(exist_ok=True)
        
        for i, page in enumerate(result.pages):
            if page.success and hasattr(page, 'raw_html') and page.raw_html:
                filename = f"page_{i:04d}_{page.url.split('/')[-1][:50]}.html"
                # Clean filename
                filename = "".join(c for c in filename if c.isalnum() or c in '.-_')
                
                html_file = site_dir / filename
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(page.raw_html)

def _display_crawl_summary(results: List[Any], conflict_keywords: List[str]):
    """Display crawl summary statistics."""
    
    if not results:
        logger.warning("No results to summarize")
        return
    
    total_sites = len(results)
    total_pages = sum(r.total_pages for r in results)
    total_successful = sum(r.successful_pages for r in results)
    total_failed = sum(r.failed_pages for r in results)
    overall_success_rate = (total_successful / total_pages * 100) if total_pages > 0 else 0
    total_time = sum(r.total_time for r in results)
    
    logger.info("\n" + "="*60)
    logger.info("📊 CRAWL SUMMARY")
    logger.info("="*60)
    logger.info(f"Sites crawled: {total_sites}")
    logger.info(f"Total pages: {total_pages}")
    logger.info(f"Successful pages: {total_successful}")
    logger.info(f"Failed pages: {total_failed}")
    logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
    logger.info(f"Total crawl time: {total_time:.1f}s")
    logger.info(f"Average pages per site: {total_pages / total_sites:.1f}")
    
    if conflict_keywords:
        total_conflict_pages = sum(
            r.site_metadata.get('conflict_analysis', {}).get('total_conflict_pages', 0)
            for r in results
        )
        conflict_percentage = (total_conflict_pages / total_pages * 100) if total_pages > 0 else 0
        
        logger.info("\n🚨 CONFLICT MONITORING")
        logger.info("-" * 30)
        logger.info(f"Conflict-related pages: {total_conflict_pages}")
        logger.info(f"Conflict percentage: {conflict_percentage:.1f}%")
        logger.info(f"Keywords monitored: {', '.join(conflict_keywords)}")
    
    # Top performing sites
    logger.info("\n🏆 TOP PERFORMING SITES")
    logger.info("-" * 30)
    sorted_results = sorted(results, key=lambda r: r.success_rate, reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        logger.info(f"{i}. {result.base_url} - {result.success_rate:.1f}% "
                   f"({result.successful_pages}/{result.total_pages} pages)")

async def analyze_command(args: argparse.Namespace):
    """Execute the analyze command."""
    
    logger.info(f"🔍 Starting analysis of: {args.input}")
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Load crawl data
    try:
        if input_path.is_file():
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Directory - look for JSON files
            json_files = list(input_path.glob("*.json"))
            if not json_files:
                logger.error(f"No JSON files found in: {args.input}")
                sys.exit(1)
            
            data = []
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Perform analysis
    analysis_results = _perform_content_analysis(data, args)
    
    # Output results
    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, default=str)
        elif args.format == 'csv':
            # Convert to CSV format
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in analysis_results.get('summary', {}).items():
                    writer.writerow([key, value])
        
        logger.info(f"Analysis results saved to: {output_path}")
    else:
        # Print to console
        print(json.dumps(analysis_results, indent=2, default=str))

def _perform_content_analysis(data: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Perform content analysis on crawl data."""
    
    analysis = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': {},
        'quality_distribution': {},
        'content_types': {},
        'site_analysis': {},
        'recommendations': []
    }
    
    if not isinstance(data, list):
        data = [data]
    
    total_pages = 0
    high_quality_pages = 0
    article_pages = 0
    sites_analyzed = set()
    
    quality_threshold = getattr(args, 'quality_threshold', 0.5)
    
    for site_result in data:
        if not isinstance(site_result, dict) or 'pages' not in site_result:
            continue
        
        base_url = site_result.get('base_url', 'unknown')
        sites_analyzed.add(base_url)
        pages = site_result.get('pages', [])
        
        site_stats = {
            'total_pages': len(pages),
            'high_quality_pages': 0,
            'article_pages': 0,
            'avg_quality_score': 0.0
        }
        
        quality_scores = []
        
        for page in pages:
            if not isinstance(page, dict):
                continue
                
            total_pages += 1
            quality_score = page.get('quality_score', 0.0)
            content_type = page.get('content_type', 'unknown')
            
            quality_scores.append(quality_score)
            
            if quality_score >= quality_threshold:
                high_quality_pages += 1
                site_stats['high_quality_pages'] += 1
            
            if content_type == 'article':
                article_pages += 1
                site_stats['article_pages'] += 1
        
        if quality_scores:
            site_stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        analysis['site_analysis'][base_url] = site_stats
    
    # Summary statistics
    analysis['summary'] = {
        'sites_analyzed': len(sites_analyzed),
        'total_pages': total_pages,
        'high_quality_pages': high_quality_pages,
        'high_quality_percentage': (high_quality_pages / total_pages * 100) if total_pages > 0 else 0,
        'article_pages': article_pages,
        'article_percentage': (article_pages / total_pages * 100) if total_pages > 0 else 0,
        'quality_threshold_used': quality_threshold
    }
    
    # Generate recommendations
    if analysis['summary']['high_quality_percentage'] < 30:
        analysis['recommendations'].append("Low quality content detected. Consider adjusting content filters.")
    
    if analysis['summary']['article_percentage'] < 20:
        analysis['recommendations'].append("Few articles found. Consider improving URL patterns or site selection.")
    
    return analysis

async def config_command(args: argparse.Namespace):
    """Execute the config command."""
    
    if args.create:
        logger.info("📝 Creating new configuration file")
        
        # Create configuration based on template
        config_manager = create_gen_config()
        
        if args.template == 'news':
            # News-optimized settings
            config_manager.settings.content_filters.include_patterns = [
                'article', 'news', 'story', 'breaking', 'report'
            ]
            config_manager.settings.content_filters.exclude_patterns = [
                'tag', 'category', 'archive', 'login', 'subscribe'
            ]
            config_manager.settings.performance.max_pages_per_site = 200
            config_manager.settings.performance.crawl_delay = 3.0
            
        elif args.template == 'research':
            # Research-optimized settings
            config_manager.settings.performance.max_pages_per_site = 1000
            config_manager.settings.content_filters.min_content_length = 500
            config_manager.settings.adaptive.enable_adaptive_crawling = True
            
        elif args.template == 'monitoring':
            # Monitoring-optimized settings
            config_manager.settings.performance.max_pages_per_site = 100
            config_manager.settings.performance.crawl_delay = 1.0
            config_manager.settings.performance.max_concurrent = 2
        
        # Save configuration
        output_path = args.output or f"gen-crawler-{args.template}-config.json"
        config_manager.save_to_file(output_path)
        
        logger.info(f"✅ Configuration created: {output_path}")
        
    elif args.validate:
        logger.info(f"🔍 Validating configuration: {args.validate}")
        
        try:
            config_manager = create_gen_config(config_file=args.validate)
            logger.info("✅ Configuration is valid")
            
            # Display configuration summary
            settings = config_manager.settings
            print(f"\nConfiguration Summary:")
            print(f"  Max pages per site: {settings.performance.max_pages_per_site}")
            print(f"  Max concurrent: {settings.performance.max_concurrent}")
            print(f"  Crawl delay: {settings.performance.crawl_delay}s")
            print(f"  Adaptive crawling: {settings.adaptive.enable_adaptive_crawling}")
            print(f"  Database enabled: {settings.database.enable_database}")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
    
    else:
        logger.error("No action specified. Use --create or --validate")
        sys.exit(1)

async def export_command(args: argparse.Namespace):
    """Execute the export command."""
    
    logger.info(f"📤 Exporting data from: {args.input}")
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Apply filters
    if hasattr(args, 'filter_quality') and args.filter_quality:
        data = _filter_by_quality(data, args.filter_quality)
        logger.info(f"Applied quality filter: >= {args.filter_quality}")
    
    if hasattr(args, 'filter_type') and args.filter_type:
        data = _filter_by_type(data, args.filter_type)
        logger.info(f"Applied type filter: {args.filter_type}")
    
    # Export in requested format
    try:
        if args.format == 'markdown':
            exporter = MarkdownExporter(
                output_dir=output_dir,
                organize_by=getattr(args, 'organize_by', 'site')
            )
            await exporter.export_results(data)
            
        elif args.format == 'html':
            exporter = HTMLExporter(output_dir=output_dir)
            await exporter.export_results(data)
            
        elif args.format == 'csv':
            exporter = CSVExporter(output_dir=output_dir)
            await exporter.export_results(data)
        
        logger.info(f"✅ Data exported to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

def _filter_by_quality(data: Any, min_quality: float) -> Any:
    """Filter data by quality score."""
    if isinstance(data, list):
        filtered_data = []
        for item in data:
            filtered_item = _filter_by_quality(item, min_quality)
            if filtered_item:
                filtered_data.append(filtered_item)
        return filtered_data
    
    elif isinstance(data, dict) and 'pages' in data:
        filtered_pages = [
            page for page in data['pages']
            if isinstance(page, dict) and page.get('quality_score', 0) >= min_quality
        ]
        if filtered_pages:
            data_copy = data.copy()
            data_copy['pages'] = filtered_pages
            return data_copy
    
    return data

def _filter_by_type(data: Any, content_type: str) -> Any:
    """Filter data by content type."""
    if isinstance(data, list):
        filtered_data = []
        for item in data:
            filtered_item = _filter_by_type(item, content_type)
            if filtered_item:
                filtered_data.append(filtered_item)
        return filtered_data
    
    elif isinstance(data, dict) and 'pages' in data:
        filtered_pages = [
            page for page in data['pages']
            if isinstance(page, dict) and page.get('content_type') == content_type
        ]
        if filtered_pages:
            data_copy = data.copy()
            data_copy['pages'] = filtered_pages
            return data_copy
    
    return data