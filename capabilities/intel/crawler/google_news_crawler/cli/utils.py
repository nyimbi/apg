#!/usr/bin/env python3
"""
Google News Crawler CLI Utilities
=================================

Utility functions for the Google News Crawler CLI.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Valid country and language codes
VALID_COUNTRIES = {
    'ET': 'Ethiopia', 'SO': 'Somalia', 'KE': 'Kenya', 'UG': 'Uganda',
    'TZ': 'Tanzania', 'SD': 'Sudan', 'SS': 'South Sudan', 'ER': 'Eritrea',
    'DJ': 'Djibouti', 'RW': 'Rwanda', 'BI': 'Burundi', 'US': 'United States',
    'GB': 'United Kingdom', 'CA': 'Canada', 'AU': 'Australia', 'DE': 'Germany',
    'FR': 'France', 'IT': 'Italy', 'ES': 'Spain', 'NL': 'Netherlands',
    'IN': 'India', 'ZA': 'South Africa', 'NG': 'Nigeria', 'EG': 'Egypt'
}

VALID_LANGUAGES = {
    'en': 'English', 'fr': 'French', 'ar': 'Arabic', 'sw': 'Swahili',
    'am': 'Amharic', 'ti': 'Tigrinya', 'om': 'Oromo', 'so': 'Somali',
    'es': 'Spanish', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
    'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'hi': 'Hindi'
}

def parse_date_input(date_str: str) -> Optional[datetime]:
    """
    Parse various date input formats.
    
    Supports:
    - YYYY-MM-DD format
    - Relative dates like "7d", "24h", "2w"
    """
    if not date_str:
        return None
    
    # Try relative date parsing first
    relative_match = re.match(r'^(\d+)([hdwmy])$', date_str.lower())
    if relative_match:
        amount, unit = relative_match.groups()
        amount = int(amount)
        
        now = datetime.now()
        if unit == 'h':  # hours
            return now - timedelta(hours=amount)
        elif unit == 'd':  # days
            return now - timedelta(days=amount)
        elif unit == 'w':  # weeks
            return now - timedelta(weeks=amount)
        elif unit == 'm':  # months (approximate)
            return now - timedelta(days=amount * 30)
        elif unit == 'y':  # years (approximate)
            return now - timedelta(days=amount * 365)
    
    # Try standard date formats
    date_formats = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d',
        '%d/%m/%Y',
        '%m/%d/%Y'
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

def validate_countries(countries: List[str]) -> List[str]:
    """Validate and filter country codes."""
    valid_countries = []
    invalid_countries = []
    
    for country in countries:
        country = country.upper().strip()
        if country in VALID_COUNTRIES:
            valid_countries.append(country)
        else:
            invalid_countries.append(country)
    
    if invalid_countries:
        logger.warning(f"Invalid country codes (ignored): {', '.join(invalid_countries)}")
        logger.info(f"Valid countries: {', '.join(VALID_COUNTRIES.keys())}")
    
    if not valid_countries:
        logger.warning("No valid countries specified, using Horn of Africa defaults")
        return ['ET', 'SO', 'KE']
    
    return valid_countries

def validate_languages(languages: List[str]) -> List[str]:
    """Validate and filter language codes."""
    valid_languages = []
    invalid_languages = []
    
    for language in languages:
        language = language.lower().strip()
        if language in VALID_LANGUAGES:
            valid_languages.append(language)
        else:
            invalid_languages.append(language)
    
    if invalid_languages:
        logger.warning(f"Invalid language codes (ignored): {', '.join(invalid_languages)}")
        logger.info(f"Valid languages: {', '.join(VALID_LANGUAGES.keys())}")
    
    if not valid_languages:
        logger.warning("No valid languages specified, using default")
        return ['en']
    
    return valid_languages

def format_output(data: Any, format_type: str = 'json') -> str:
    """Format data for output."""
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    elif format_type == 'pretty':
        # Simple pretty printing for complex data
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{key}: {json.dumps(value, default=str)}")
                else:
                    lines.append(f"{key}: {value}")
            return '\n'.join(lines)
        elif isinstance(data, list):
            return '\n'.join(str(item) for item in data)
        else:
            return str(data)
    else:
        return str(data)

def load_cli_config(config_path: Path) -> Dict[str, Any]:
    """Load CLI configuration from file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def save_cli_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save CLI configuration to file."""
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save configuration: {e}")

def create_mock_db_manager():
    """Create a mock database manager for CLI operations."""
    
    class MockDBManager:
        """Mock database manager for CLI testing and demonstration."""
        
        def __init__(self):
            self.stored_articles = []
        
        async def store_articles(self, articles: List[Dict[str, Any]]) -> int:
            """Mock method to store articles."""
            self.stored_articles.extend(articles)
            logger.debug(f"Mock stored {len(articles)} articles")
            return len(articles)
        
        async def get_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
            """Mock method to retrieve articles."""
            return self.stored_articles[-limit:]
        
        async def close(self):
            """Mock close method."""
            pass
        
        def __str__(self):
            return f"MockDBManager(stored={len(self.stored_articles)} articles)"
    
    return MockDBManager()

def parse_query_operators(query: str) -> Dict[str, Any]:
    """Parse query for special operators and filters."""
    operators = {
        'include_terms': [],
        'exclude_terms': [],
        'exact_phrases': [],
        'any_terms': [],
        'original_query': query
    }
    
    # Extract quoted phrases (exact matches)
    phrase_pattern = r'"([^"]*)"'
    phrases = re.findall(phrase_pattern, query)
    operators['exact_phrases'] = phrases
    
    # Remove phrases from query for further processing
    query_without_phrases = re.sub(phrase_pattern, '', query)
    
    # Extract exclusions (terms with -)
    exclusion_pattern = r'-(\w+)'
    exclusions = re.findall(exclusion_pattern, query_without_phrases)
    operators['exclude_terms'] = exclusions
    
    # Remove exclusions from query
    query_without_exclusions = re.sub(exclusion_pattern, '', query_without_phrases)
    
    # Extract OR groups
    or_pattern = r'\(([^)]*)\)'
    or_groups = re.findall(or_pattern, query_without_exclusions)
    for group in or_groups:
        operators['any_terms'].extend([term.strip() for term in group.split(' OR ')])
    
    # Remove OR groups from query
    query_remaining = re.sub(or_pattern, '', query_without_exclusions)
    
    # Remaining terms are include terms
    include_terms = [term.strip() for term in query_remaining.split() if term.strip()]
    operators['include_terms'] = include_terms
    
    return operators

def validate_file_path(filepath: str, must_exist: bool = False) -> Path:
    """Validate and normalize file path."""
    path = Path(filepath).expanduser().resolve()
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    if not must_exist:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path

def estimate_processing_time(num_articles: int, enable_crawlee: bool) -> Tuple[int, str]:
    """Estimate processing time for articles."""
    if enable_crawlee:
        # Crawlee processing: ~2-5 seconds per article
        seconds = num_articles * 3.5
    else:
        # RSS only: ~0.1-0.5 seconds per article
        seconds = num_articles * 0.3
    
    if seconds < 60:
        return int(seconds), f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return int(seconds), f"{seconds/60:.1f} minutes"
    else:
        return int(seconds), f"{seconds/3600:.1f} hours"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def colorize_output(text: str, color: str = 'default') -> str:
    """Add color codes to text for terminal output."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'end': '\033[0m',
        'default': ''
    }
    
    if color in colors and color != 'default':
        return f"{colors[color]}{text}{colors['end']}"
    return text

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a simple progress bar."""
    if total == 0:
        return "[" + " " * width + "] 0%"
    
    percentage = current / total
    filled = int(width * percentage)
    bar = "█" * filled + "░" * (width - filled)
    
    return f"[{bar}] {percentage*100:.1f}% ({current}/{total})"

def validate_url(url: str) -> bool:
    """Validate URL format."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    # Limit length
    sanitized = sanitized[:200]
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized

def parse_size_input(size_str: str) -> int:
    """Parse size input like '10MB', '500KB', etc."""
    if not size_str:
        return 0
    
    size_str = size_str.upper().strip()
    
    # Extract number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$', size_str)
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    
    number, unit = match.groups()
    number = float(number)
    
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4,
        '': 1  # No unit defaults to bytes
    }
    
    return int(number * multipliers.get(unit, 1))

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return "unknown"

# Export all utility functions
__all__ = [
    'parse_date_input',
    'validate_countries',
    'validate_languages',
    'format_output',
    'load_cli_config',
    'save_cli_config',
    'create_mock_db_manager',
    'parse_query_operators',
    'validate_file_path',
    'estimate_processing_time',
    'format_file_size',
    'truncate_text',
    'colorize_output',
    'create_progress_bar',
    'validate_url',
    'sanitize_filename',
    'parse_size_input',
    'format_duration',
    'extract_domain'
]