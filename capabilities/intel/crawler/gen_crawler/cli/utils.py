"""
Gen Crawler CLI Utilities
=========================

Utility functions for the CLI interface including logging setup,
URL validation, and result formatting.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from urllib.parse import urlparse
import json
from datetime import datetime

def setup_logging(verbose: int = 0, quiet: bool = False, log_file: Optional[str] = None) -> int:
    """
    Setup logging configuration based on CLI arguments.
    
    Args:
        verbose: Verbosity level (0-3)
        quiet: Whether to suppress output
        log_file: Optional log file path
        
    Returns:
        Configured log level
    """
    
    # Determine log level
    if quiet:
        log_level = logging.ERROR
    elif verbose == 0:
        log_level = logging.INFO
    elif verbose == 1:
        log_level = logging.DEBUG
    else:  # verbose >= 2
        log_level = logging.DEBUG
    
    # Create formatters
    if verbose >= 2:
        # Detailed format for high verbosity
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
    elif verbose == 1:
        # Medium detail format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
    else:
        # Simple format for normal use
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always debug level for files
            
            # Use detailed format for file logging
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}", file=sys.stderr)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Special handling for Crawlee/Playwright loggers
    logging.getLogger('playwright').setLevel(logging.WARNING)
    logging.getLogger('crawlee').setLevel(logging.INFO if verbose >= 1 else logging.WARNING)
    
    return log_level

def validate_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of URLs.
    
    Args:
        urls: List of URL strings to validate
        
    Returns:
        Tuple of (valid_urls, invalid_urls)
    """
    
    valid_urls = []
    invalid_urls = []
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    for url in urls:
        url = url.strip()
        
        if not url:
            continue
        
        # Add https:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Validate URL format
        if url_pattern.match(url):
            try:
                parsed = urlparse(url)
                if parsed.netloc and parsed.scheme in ('http', 'https'):
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            except Exception:
                invalid_urls.append(url)
        else:
            invalid_urls.append(url)
    
    return valid_urls, invalid_urls

def format_results(results: List[Any], format_type: str = 'summary') -> str:
    """
    Format crawl results for display.
    
    Args:
        results: List of crawl results
        format_type: Type of formatting ('summary', 'detailed', 'json')
        
    Returns:
        Formatted string
    """
    
    if not results:
        return "No results to display."
    
    if format_type == 'json':
        return json.dumps(results, indent=2, default=str)
    
    output_lines = []
    
    if format_type == 'summary':
        output_lines.append("CRAWL RESULTS SUMMARY")
        output_lines.append("=" * 50)
        
        total_sites = len(results)
        total_pages = sum(r.get('total_pages', 0) for r in results if isinstance(r, dict))
        total_successful = sum(r.get('successful_pages', 0) for r in results if isinstance(r, dict))
        
        output_lines.append(f"Sites crawled: {total_sites}")
        output_lines.append(f"Total pages: {total_pages:,}")
        output_lines.append(f"Successful pages: {total_successful:,}")
        
        if total_pages > 0:
            success_rate = (total_successful / total_pages) * 100
            output_lines.append(f"Overall success rate: {success_rate:.1f}%")
        
        output_lines.append("")
        
        # Site breakdown
        output_lines.append("SITE BREAKDOWN")
        output_lines.append("-" * 30)
        
        for i, result in enumerate(results, 1):
            if not isinstance(result, dict):
                continue
            
            base_url = result.get('base_url', f'Site {i}')
            total = result.get('total_pages', 0)
            successful = result.get('successful_pages', 0)
            rate = (successful / total * 100) if total > 0 else 0
            
            output_lines.append(f"{i}. {base_url}")
            output_lines.append(f"   Pages: {successful}/{total} ({rate:.1f}%)")
    
    elif format_type == 'detailed':
        for i, result in enumerate(results, 1):
            if not isinstance(result, dict):
                continue
            
            output_lines.append(f"SITE {i}: {result.get('base_url', 'Unknown')}")
            output_lines.append("=" * 60)
            
            # Basic stats
            output_lines.append(f"Total pages: {result.get('total_pages', 0):,}")
            output_lines.append(f"Successful: {result.get('successful_pages', 0):,}")
            output_lines.append(f"Failed: {result.get('failed_pages', 0):,}")
            output_lines.append(f"Success rate: {result.get('success_rate', 0):.1f}%")
            output_lines.append(f"Crawl time: {result.get('total_time', 0):.1f}s")
            
            # Sample pages
            pages = result.get('pages', [])
            if pages:
                output_lines.append(f"\nSample pages (first 5):")
                for j, page in enumerate(pages[:5], 1):
                    if isinstance(page, dict):
                        title = page.get('title', 'Untitled')[:50]
                        url = page.get('url', '')[:60]
                        word_count = page.get('word_count', 0)
                        content_type = page.get('content_type', 'unknown')
                        
                        output_lines.append(f"  {j}. {title}")
                        output_lines.append(f"     URL: {url}")
                        output_lines.append(f"     Type: {content_type}, Words: {word_count:,}")
            
            output_lines.append("")
    
    return "\n".join(output_lines)

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a text progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        width: Width of progress bar in characters
        
    Returns:
        Progress bar string
    """
    
    if total <= 0:
        return f"[{'=' * width}] 0/0"
    
    percentage = min(current / total, 1.0)
    filled_width = int(width * percentage)
    bar = '=' * filled_width + '-' * (width - filled_width)
    
    return f"[{bar}] {current}/{total} ({percentage * 100:.1f}%)"

def sanitize_path(path: str) -> str:
    """
    Sanitize a path string for filesystem use.
    
    Args:
        path: Path string to sanitize
        
    Returns:
        Sanitized path string
    """
    
    # Replace invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        path = path.replace(char, '_')
    
    # Remove or replace other problematic characters
    path = path.replace('/', '_')
    path = path.replace('\\', '_')
    path = re.sub(r'\s+', '_', path)  # Replace whitespace with underscores
    path = re.sub(r'[^\w\-_.]', '', path)  # Keep only word chars, hyphens, underscores, dots
    
    # Remove leading/trailing dots and spaces
    path = path.strip('. ')
    
    # Ensure it's not empty
    if not path:
        path = 'unnamed'
    
    return path

def load_proxy_list(proxy_file: str) -> List[str]:
    """
    Load proxy list from file.
    
    Args:
        proxy_file: Path to proxy file
        
    Returns:
        List of proxy URLs
    """
    
    try:
        proxy_path = Path(proxy_file)
        if not proxy_path.exists():
            return []
        
        with open(proxy_path, 'r', encoding='utf-8') as f:
            proxies = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Basic proxy format validation
                    if ':' in line:
                        proxies.append(line)
        
        return proxies
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not load proxy list: {e}")
        return []

def estimate_crawl_time(num_pages: int, crawl_delay: float, max_concurrent: int) -> float:
    """
    Estimate crawl time based on parameters.
    
    Args:
        num_pages: Number of pages to crawl
        crawl_delay: Delay between requests in seconds
        max_concurrent: Maximum concurrent requests
        
    Returns:
        Estimated time in seconds
    """
    
    if num_pages <= 0 or max_concurrent <= 0:
        return 0.0
    
    # Basic estimation: pages / concurrency * delay
    # Add some overhead for connection setup, parsing, etc.
    base_time = (num_pages / max_concurrent) * crawl_delay
    overhead = num_pages * 0.5  # 0.5 seconds overhead per page
    
    return base_time + overhead

def create_summary_report(results: List[Any]) -> Dict[str, Any]:
    """
    Create a comprehensive summary report from crawl results.
    
    Args:
        results: List of crawl results
        
    Returns:
        Summary report dictionary
    """
    
    if not results:
        return {'error': 'No results to summarize'}
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'overview': {},
        'performance': {},
        'content_analysis': {},
        'quality_metrics': {},
        'recommendations': []
    }
    
    # Overview statistics
    total_sites = len(results)
    total_pages = sum(r.get('total_pages', 0) for r in results if isinstance(r, dict))
    total_successful = sum(r.get('successful_pages', 0) for r in results if isinstance(r, dict))
    total_failed = sum(r.get('failed_pages', 0) for r in results if isinstance(r, dict))
    total_time = sum(r.get('total_time', 0) for r in results if isinstance(r, dict))
    
    report['overview'] = {
        'sites_crawled': total_sites,
        'total_pages': total_pages,
        'successful_pages': total_successful,
        'failed_pages': total_failed,
        'overall_success_rate': (total_successful / total_pages * 100) if total_pages > 0 else 0,
        'total_crawl_time': total_time
    }
    
    # Performance metrics
    if total_time > 0:
        pages_per_second = total_pages / total_time
        avg_time_per_page = total_time / total_pages if total_pages > 0 else 0
        
        report['performance'] = {
            'pages_per_second': pages_per_second,
            'average_time_per_page': avg_time_per_page,
            'average_time_per_site': total_time / total_sites if total_sites > 0 else 0
        }
    
    # Content analysis
    all_pages = []
    for result in results:
        if isinstance(result, dict) and 'pages' in result:
            all_pages.extend(result['pages'])
    
    if all_pages:
        content_types = {}
        quality_scores = []
        word_counts = []
        
        for page in all_pages:
            if isinstance(page, dict):
                content_type = page.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                quality_score = page.get('quality_score', 0)
                if quality_score > 0:
                    quality_scores.append(quality_score)
                
                word_count = page.get('word_count', 0)
                if word_count > 0:
                    word_counts.append(word_count)
        
        report['content_analysis'] = {
            'content_types': content_types,
            'total_words': sum(word_counts),
            'average_words_per_page': sum(word_counts) / len(word_counts) if word_counts else 0
        }
        
        if quality_scores:
            report['quality_metrics'] = {
                'average_quality_score': sum(quality_scores) / len(quality_scores),
                'high_quality_pages': sum(1 for score in quality_scores if score > 0.7),
                'low_quality_pages': sum(1 for score in quality_scores if score < 0.3)
            }
    
    # Generate recommendations
    success_rate = report['overview']['overall_success_rate']
    if success_rate < 50:
        report['recommendations'].append(
            "Low success rate detected. Consider adjusting crawler settings or target sites."
        )
    elif success_rate < 80:
        report['recommendations'].append(
            "Moderate success rate. Fine-tuning crawler parameters may improve results."
        )
    
    avg_quality = report.get('quality_metrics', {}).get('average_quality_score', 0)
    if avg_quality < 0.5:
        report['recommendations'].append(
            "Low content quality detected. Consider improving content filtering and extraction."
        )
    
    if total_time > 0:
        pages_per_second = report['performance']['pages_per_second']
        if pages_per_second < 0.1:
            report['recommendations'].append(
                "Slow crawling speed. Consider increasing concurrency or reducing delays."
            )
    
    return report