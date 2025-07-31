#!/usr/bin/env python3
"""
Target Locations Configuration Example
======================================

Configuration examples for the enhanced search crawler with stealth capabilities
for monitoring conflicts in specific target locations (Aweil, Karamoja, Mandera, Assosa).

Author: Lindela Development Team
"""

from typing import Dict, Any, List

def get_production_config() -> Dict[str, Any]:
    """
    Production configuration for target locations monitoring.
    Optimized for comprehensive coverage with stealth capabilities.
    """
    return {
        # Target locations
        'target_locations': ['Aweil', 'Karamoja', 'Mandera', 'Assosa'],
        'min_articles_per_location': 50,
        'enable_hierarchical_search': True,
        
        # Stealth configuration
        'enable_stealth_download': True,
        'stealth_aggressive': False,  # Conservative for production
        'use_stealth': True,
        'rotate_user_agents': True,
        
        # Search engines - all 10 for maximum coverage
        'engines': [
            'google', 'bing', 'duckduckgo', 'yandex', 'brave', 
            'startpage', 'searx', 'mojeek', 'swisscows', 'baidu'
        ],
        'max_results_per_engine': 10,
        'total_max_results': 500,  # 50 per location * 4 locations * buffer
        'parallel_searches': True,
        'timeout': 60.0,
        
        # Content downloading
        'download_content': True,
        'parse_content': True,
        'extract_metadata': True,
        'max_concurrent_downloads': 3,  # Conservative for stealth
        
        # Caching for efficiency
        'enable_cache': True,
        'cache_ttl': 7200,  # 2 hours
        
        # Rate limiting for stealth
        'min_delay_between_searches': 2.0,
        'use_proxies': False,  # Enable when proxy pool is available
        
        # Conflict analysis
        'extract_entities': True,
        'analyze_sentiment': True,
        'detect_locations': True,
        'track_developments': True,
        'enable_alerts': True,
        
        # Scoring optimization
        'min_relevance_score': 0.5,
        'escalation_threshold': 0.7,
        'location_weight': 0.3,
        'temporal_weight': 0.25,
        'keyword_weight': 0.25,
        'source_weight': 0.2,
        
        # Trusted sources for credibility scoring
        'trusted_sources': [
            'reuters.com', 'apnews.com', 'bbc.com', 'aljazeera.com',
            'france24.com', 'dw.com', 'africanews.com', 'bloomberg.com',
            'cnn.com', 'theguardian.com', 'washingtonpost.com', 'nytimes.com',
            'standardmedia.co.ke', 'monitor.co.ug', 'theeastafrican.co.ke',
            'sudantribune.com', 'hiiraan.com', 'garowe-online.com'
        ],
        
        # Alert configuration
        'critical_keywords': [
            'casualties', 'killed', 'explosion', 'attack', 'invasion',
            'emergency', 'crisis', 'escalation', 'violence', 'massacre',
            'al-shabaab', 'displacement', 'refugee', 'ethnic violence'
        ],
        
        # Time filtering
        'max_age_days': 7,
        
        # Monitoring regions
        'conflict_regions': ['horn_of_africa'],
        'monitor_keywords': [
            'conflict', 'violence', 'attack', 'killed', 'displaced',
            'cattle raiding', 'ethnic violence', 'cross-border',
            'al-shabaab', 'pastoral conflict', 'border security'
        ]
    }


def get_development_config() -> Dict[str, Any]:
    """
    Development/testing configuration with reduced load.
    """
    return {
        # Target locations
        'target_locations': ['Aweil', 'Mandera'],  # Reduced for testing
        'min_articles_per_location': 20,
        'enable_hierarchical_search': True,
        
        # Stealth configuration - less aggressive for development
        'enable_stealth_download': True,
        'stealth_aggressive': False,
        'use_stealth': True,
        'rotate_user_agents': True,
        
        # Limited search engines for faster testing
        'engines': ['google', 'bing', 'duckduckgo', 'brave'],
        'max_results_per_engine': 5,
        'total_max_results': 50,
        'parallel_searches': True,
        'timeout': 30.0,
        
        # Content downloading
        'download_content': True,
        'parse_content': True,
        'max_concurrent_downloads': 2,
        
        # Minimal caching for development
        'enable_cache': True,
        'cache_ttl': 1800,  # 30 minutes
        
        # Faster rate limiting for development
        'min_delay_between_searches': 1.0,
        
        # Basic analysis
        'extract_entities': True,
        'analyze_sentiment': False,  # Disable for faster testing
        'detect_locations': True,
        'track_developments': False,
        'enable_alerts': True,
        
        # Relaxed scoring
        'min_relevance_score': 0.3,
        'escalation_threshold': 0.6,
        
        # Basic trusted sources
        'trusted_sources': [
            'reuters.com', 'bbc.com', 'aljazeera.com', 'africanews.com'
        ],
        
        # Time filtering
        'max_age_days': 3,
        
        # Basic monitoring
        'conflict_regions': ['horn_of_africa'],
        'monitor_keywords': ['conflict', 'violence', 'attack']
    }


def get_stealth_optimized_config() -> Dict[str, Any]:
    """
    Configuration optimized for stealth operation on protected sites.
    """
    return {
        # Target locations
        'target_locations': ['Aweil', 'Karamoja', 'Mandera', 'Assosa'],
        'min_articles_per_location': 50,
        'enable_hierarchical_search': True,
        
        # Aggressive stealth configuration
        'enable_stealth_download': True,
        'stealth_aggressive': True,  # Aggressive stealth
        'use_stealth': True,
        'rotate_user_agents': True,
        'use_proxies': True,  # Enable when proxy pool is available
        
        # Conservative search engines for stealth
        'engines': ['duckduckgo', 'startpage', 'searx', 'brave', 'yandex'],
        'max_results_per_engine': 15,
        'total_max_results': 300,
        'parallel_searches': False,  # Sequential for stealth
        'timeout': 90.0,  # Longer timeout for stealth operations
        
        # Conservative downloading
        'download_content': True,
        'parse_content': True,
        'max_concurrent_downloads': 1,  # Very conservative
        
        # Extended caching
        'enable_cache': True,
        'cache_ttl': 14400,  # 4 hours
        
        # Aggressive rate limiting
        'min_delay_between_searches': 5.0,  # 5 second delays
        
        # Full analysis enabled
        'extract_entities': True,
        'analyze_sentiment': True,
        'detect_locations': True,
        'track_developments': True,
        'enable_alerts': True,
        
        # Standard scoring
        'min_relevance_score': 0.5,
        'escalation_threshold': 0.7,
        
        # Extended trusted sources
        'trusted_sources': [
            'reuters.com', 'apnews.com', 'bbc.com', 'aljazeera.com',
            'france24.com', 'dw.com', 'africanews.com', 'bloomberg.com',
            'standardmedia.co.ke', 'monitor.co.ug', 'theeastafrican.co.ke'
        ],
        
        # Time filtering
        'max_age_days': 7,
        
        # Comprehensive monitoring
        'conflict_regions': ['horn_of_africa'],
        'monitor_keywords': [
            'conflict', 'violence', 'attack', 'killed', 'displaced',
            'cattle raiding', 'ethnic violence', 'al-shabaab',
            'pastoral conflict', 'border security', 'cross-border'
        ]
    }


def get_high_performance_config() -> Dict[str, Any]:
    """
    Configuration optimized for maximum speed and coverage.
    Less stealth, more aggressive parallel processing.
    """
    return {
        # Target locations
        'target_locations': ['Aweil', 'Karamoja', 'Mandera', 'Assosa'],
        'min_articles_per_location': 75,  # Higher target
        'enable_hierarchical_search': True,
        
        # Basic stealth only
        'enable_stealth_download': True,
        'stealth_aggressive': False,
        'use_stealth': True,
        'rotate_user_agents': True,
        
        # All search engines for maximum coverage
        'engines': [
            'google', 'bing', 'duckduckgo', 'yandex', 'brave', 
            'startpage', 'searx', 'mojeek', 'swisscows', 'baidu'
        ],
        'max_results_per_engine': 15,
        'total_max_results': 750,  # 75 per location * 4 locations * buffer
        'parallel_searches': True,  # Maximum parallelism
        'timeout': 45.0,
        
        # Aggressive downloading
        'download_content': True,
        'parse_content': True,
        'max_concurrent_downloads': 5,  # More concurrent
        
        # Minimal caching for freshness
        'enable_cache': True,
        'cache_ttl': 3600,  # 1 hour
        
        # Minimal delays for speed
        'min_delay_between_searches': 0.5,
        
        # Full analysis
        'extract_entities': True,
        'analyze_sentiment': True,
        'detect_locations': True,
        'track_developments': True,
        'enable_alerts': True,
        
        # Standard scoring
        'min_relevance_score': 0.4,  # Lower threshold for more coverage
        'escalation_threshold': 0.7,
        
        # Comprehensive trusted sources
        'trusted_sources': [
            'reuters.com', 'apnews.com', 'bbc.com', 'aljazeera.com',
            'france24.com', 'dw.com', 'africanews.com', 'bloomberg.com',
            'cnn.com', 'theguardian.com', 'washingtonpost.com',
            'standardmedia.co.ke', 'monitor.co.ug', 'theeastafrican.co.ke',
            'sudantribune.com', 'hiiraan.com', 'garowe-online.com'
        ],
        
        # Time filtering
        'max_age_days': 14,  # Broader time range
        
        # Comprehensive monitoring
        'conflict_regions': ['horn_of_africa'],
        'monitor_keywords': [
            'conflict', 'violence', 'attack', 'killed', 'displaced',
            'cattle raiding', 'ethnic violence', 'al-shabaab',
            'pastoral conflict', 'border security', 'drought',
            'refugee', 'displacement', 'crisis', 'emergency'
        ]
    }


# Location-specific keyword configurations
TARGET_LOCATION_KEYWORDS = {
    'Aweil': [
        'Aweil', 'Northern Bahr el Ghazal', 'South Sudan',
        'cattle raiding', 'Dinka', 'Arab tribes', 'displacement',
        'Wau', 'Bentiu', 'ethnic violence', 'pastoral conflict'
    ],
    'Karamoja': [
        'Karamoja', 'Kotido', 'Moroto', 'Kaabong', 'Uganda',
        'cattle rustling', 'Karimojong', 'disarmament', 'pastoral',
        'Turkana', 'drought conflict', 'warrior culture'
    ],
    'Mandera': [
        'Mandera', 'Kenya', 'Al-Shabaab', 'cross-border',
        'Somalia border', 'Ethiopia border', 'clan violence',
        'Banisa', 'Lafey', 'pastoralist conflict', 'security'
    ],
    'Assosa': [
        'Assosa', 'Benishangul-Gumuz', 'Ethiopia', 'ethnic violence',
        'Gumuz', 'Oromo', 'displacement', 'Sudan border',
        'Blue Nile', 'GERD', 'land disputes', 'Metekel'
    ]
}


def get_location_specific_config(location: str) -> Dict[str, Any]:
    """
    Get configuration optimized for a specific target location.
    
    Args:
        location: Target location (Aweil, Karamoja, Mandera, or Assosa)
        
    Returns:
        Configuration dictionary
    """
    base_config = get_production_config()
    
    if location in TARGET_LOCATION_KEYWORDS:
        base_config.update({
            'target_locations': [location],
            'min_articles_per_location': 100,  # Higher for single location
            'monitor_keywords': TARGET_LOCATION_KEYWORDS[location],
            'total_max_results': 150  # More focused search
        })
    
    return base_config


# Usage examples
if __name__ == "__main__":
    print("Target Locations Configuration Examples")
    print("=" * 50)
    
    configs = {
        'production': get_production_config(),
        'development': get_development_config(),
        'stealth': get_stealth_optimized_config(),
        'performance': get_high_performance_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} CONFIG:")
        print(f"  Target Locations: {config['target_locations']}")
        print(f"  Min Articles/Location: {config['min_articles_per_location']}")
        print(f"  Search Engines: {len(config['engines'])}")
        print(f"  Stealth Enabled: {config['enable_stealth_download']}")
        print(f"  Stealth Aggressive: {config.get('stealth_aggressive', False)}")
        print(f"  Max Results: {config['total_max_results']}")
        print(f"  Parallel Searches: {config['parallel_searches']}")
        print(f"  Rate Limit: {config['min_delay_between_searches']}s")
    
    print(f"\nAWEIL-SPECIFIC CONFIG:")
    aweil_config = get_location_specific_config('Aweil')
    print(f"  Keywords: {aweil_config['monitor_keywords']}")
    print(f"  Min Articles: {aweil_config['min_articles_per_location']}")