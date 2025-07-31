"""
Adaptive Crawler Strategy Management
===================================

Advanced crawler strategy management for optimizing crawling behavior
based on site characteristics and performance patterns.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class CrawlStrategy(Enum):
    """Available crawling strategies."""
    ADAPTIVE = "adaptive"          # Let AdaptivePlaywrightCrawler decide
    HTTP_ONLY = "http_only"        # Force HTTP-only crawling
    BROWSER_ONLY = "browser_only"  # Force browser-based crawling
    MIXED = "mixed"                # Alternate between strategies

@dataclass
class SiteProfile:
    """Profile information for a crawled site."""
    domain: str
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    average_load_time: float = 0.0
    last_crawled: Optional[datetime] = None
    preferred_strategy: CrawlStrategy = CrawlStrategy.ADAPTIVE
    robots_txt_respected: bool = True
    requires_javascript: bool = False
    has_infinite_scroll: bool = False
    cloudflare_protection: bool = False
    rate_limit_detected: bool = False
    performance_score: float = 0.0
    content_quality_score: float = 0.0
    site_characteristics: Dict[str, Any] = field(default_factory=dict)
    crawl_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100
    
    def update_performance(self, pages_crawled: int, successful: int, load_time: float):
        """Update performance metrics."""
        self.total_pages += pages_crawled
        self.successful_pages += successful
        self.failed_pages += (pages_crawled - successful)
        
        # Update average load time
        if self.average_load_time == 0:
            self.average_load_time = load_time
        else:
            self.average_load_time = (self.average_load_time + load_time) / 2
        
        self.last_crawled = datetime.now()
        
        # Calculate performance score
        self._calculate_performance_score()
    
    def _calculate_performance_score(self):
        """Calculate overall performance score."""
        success_weight = 0.4
        speed_weight = 0.3
        reliability_weight = 0.3
        
        # Success rate component (0-100)
        success_component = self.success_rate
        
        # Speed component (inverse of load time, normalized)
        speed_component = max(0, 100 - (self.average_load_time * 10))
        
        # Reliability component (based on consistent performance)
        reliability_component = 100 if not self.rate_limit_detected else 50
        
        self.performance_score = (
            success_component * success_weight +
            speed_component * speed_weight +
            reliability_component * reliability_weight
        )
    
    def add_crawl_record(self, result: Dict[str, Any]):
        """Add a crawl record to history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'pages_crawled': result.get('pages_crawled', 0),
            'success_rate': result.get('success_rate', 0),
            'load_time': result.get('average_load_time', 0),
            'strategy_used': result.get('strategy', 'unknown')
        }
        self.crawl_history.append(record)
        
        # Keep only last 10 records
        if len(self.crawl_history) > 10:
            self.crawl_history = self.crawl_history[-10:]

class AdaptiveCrawler:
    """
    Adaptive crawler strategy manager that optimizes crawling behavior
    based on site characteristics and historical performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adaptive crawler manager."""
        self.config = config or {}
        self.site_profiles: Dict[str, SiteProfile] = {}
        self.global_stats = {
            'sites_analyzed': 0,
            'total_crawls': 0,
            'strategy_performance': {
                'adaptive': {'count': 0, 'avg_success': 0.0},
                'http_only': {'count': 0, 'avg_success': 0.0},
                'browser_only': {'count': 0, 'avg_success': 0.0},
                'mixed': {'count': 0, 'avg_success': 0.0}
            }
        }
    
    def get_site_profile(self, url: str) -> SiteProfile:
        """Get or create site profile for a URL."""
        domain = urlparse(url).netloc
        
        if domain not in self.site_profiles:
            self.site_profiles[domain] = SiteProfile(domain=domain)
            self.global_stats['sites_analyzed'] += 1
        
        return self.site_profiles[domain]
    
    def recommend_strategy(self, url: str, context: Optional[Dict[str, Any]] = None) -> CrawlStrategy:
        """
        Recommend the best crawling strategy for a given URL.
        
        Args:
            url: Target URL
            context: Additional context information
            
        Returns:
            Recommended CrawlStrategy
        """
        profile = self.get_site_profile(url)
        context = context or {}
        
        # For new sites, start with adaptive strategy
        if profile.total_pages == 0:
            return CrawlStrategy.ADAPTIVE
        
        # If site has been successfully crawled recently, use preferred strategy
        if profile.last_crawled and profile.success_rate > 80:
            recent_threshold = datetime.now() - timedelta(hours=24)
            if profile.last_crawled > recent_threshold:
                return profile.preferred_strategy
        
        # Analyze site characteristics
        strategy = self._analyze_site_characteristics(profile, context)
        
        # Update preferred strategy if performance is good
        if profile.success_rate > 85:
            profile.preferred_strategy = strategy
        
        return strategy
    
    def _analyze_site_characteristics(self, profile: SiteProfile, context: Dict[str, Any]) -> CrawlStrategy:
        """Analyze site characteristics to determine best strategy."""
        
        # If site requires JavaScript or has infinite scroll, prefer browser-based
        if profile.requires_javascript or profile.has_infinite_scroll:
            return CrawlStrategy.BROWSER_ONLY
        
        # If site has Cloudflare protection, use adaptive
        if profile.cloudflare_protection:
            return CrawlStrategy.ADAPTIVE
        
        # If rate limiting detected, slow down with HTTP-only
        if profile.rate_limit_detected:
            return CrawlStrategy.HTTP_ONLY
        
        # For high-performance sites with simple content, use HTTP-only
        if profile.success_rate > 90 and profile.average_load_time < 2.0:
            return CrawlStrategy.HTTP_ONLY
        
        # Default to adaptive for balanced approach
        return CrawlStrategy.ADAPTIVE
    
    def update_strategy_performance(self, url: str, strategy: CrawlStrategy, 
                                  success_rate: float, load_time: float):
        """Update performance metrics for a strategy."""
        profile = self.get_site_profile(url)
        
        # Update site profile
        pages_crawled = 1  # Simplified for this update
        successful = 1 if success_rate > 50 else 0
        profile.update_performance(pages_crawled, successful, load_time)
        
        # Update global strategy performance
        strategy_key = strategy.value
        if strategy_key in self.global_stats['strategy_performance']:
            stats = self.global_stats['strategy_performance'][strategy_key]
            stats['count'] += 1
            
            # Update average success rate
            if stats['count'] == 1:
                stats['avg_success'] = success_rate
            else:
                stats['avg_success'] = (stats['avg_success'] + success_rate) / 2
        
        self.global_stats['total_crawls'] += 1
    
    def detect_site_features(self, url: str, page_content: str, 
                           load_time: float) -> Dict[str, Any]:
        """
        Detect site features that affect crawling strategy.
        
        Args:
            url: Page URL
            page_content: HTML content
            load_time: Page load time
            
        Returns:
            Dictionary of detected features
        """
        features = {
            'requires_javascript': False,
            'has_infinite_scroll': False,
            'cloudflare_protection': False,
            'content_quality': 'unknown',
            'estimated_total_pages': 0
        }
        
        if not page_content:
            return features
        
        content_lower = page_content.lower()
        
        # Detect JavaScript requirements
        js_indicators = [
            'document.addeventlistener',
            'window.onload',
            'react', 'angular', 'vue',
            'spa-content', 'dynamic-content'
        ]
        features['requires_javascript'] = any(indicator in content_lower for indicator in js_indicators)
        
        # Detect infinite scroll
        scroll_indicators = [
            'infinite-scroll', 'lazy-load', 'load-more',
            'pagination-ajax', 'scroll-trigger'
        ]
        features['has_infinite_scroll'] = any(indicator in content_lower for indicator in scroll_indicators)
        
        # Detect Cloudflare protection
        cf_indicators = [
            'cloudflare', 'cf-ray', 'checking your browser',
            'ddos protection', 'security check'
        ]
        features['cloudflare_protection'] = any(indicator in content_lower for indicator in cf_indicators)
        
        # Estimate content quality
        if len(page_content) > 5000 and 'article' in content_lower:
            features['content_quality'] = 'high'
        elif len(page_content) > 1000:
            features['content_quality'] = 'medium'
        else:
            features['content_quality'] = 'low'
        
        # Update site profile with detected features
        profile = self.get_site_profile(url)
        profile.requires_javascript = features['requires_javascript']
        profile.has_infinite_scroll = features['has_infinite_scroll']
        profile.cloudflare_protection = features['cloudflare_protection']
        profile.site_characteristics.update(features)
        
        return features
    
    def get_optimal_settings(self, url: str, strategy: CrawlStrategy) -> Dict[str, Any]:
        """
        Get optimal crawler settings for a URL and strategy.
        
        Args:
            url: Target URL
            strategy: Chosen strategy
            
        Returns:
            Dictionary of optimal settings
        """
        profile = self.get_site_profile(url)
        
        base_settings = {
            'max_concurrent': 3,
            'request_delay': 2.0,
            'request_timeout': 30,
            'max_retries': 3,
            'respect_robots_txt': True
        }
        
        # Adjust based on strategy
        if strategy == CrawlStrategy.HTTP_ONLY:
            base_settings.update({
                'max_concurrent': 5,
                'request_delay': 1.0,
                'request_timeout': 15
            })
        elif strategy == CrawlStrategy.BROWSER_ONLY:
            base_settings.update({
                'max_concurrent': 2,
                'request_delay': 3.0,
                'request_timeout': 60
            })
        
        # Adjust based on site characteristics
        if profile.rate_limit_detected:
            base_settings['request_delay'] *= 2
            base_settings['max_concurrent'] = min(base_settings['max_concurrent'], 2)
        
        if profile.cloudflare_protection:
            base_settings['request_delay'] = max(base_settings['request_delay'], 3.0)
            base_settings['max_retries'] = 5
        
        return base_settings
    
    def get_crawler_stats(self) -> Dict[str, Any]:
        """Get comprehensive crawler statistics."""
        return {
            'global_stats': self.global_stats,
            'site_profiles': {
                domain: {
                    'success_rate': profile.success_rate,
                    'performance_score': profile.performance_score,
                    'preferred_strategy': profile.preferred_strategy.value,
                    'total_pages': profile.total_pages,
                    'last_crawled': profile.last_crawled.isoformat() if profile.last_crawled else None
                }
                for domain, profile in self.site_profiles.items()
            },
            'strategy_recommendations': self._get_strategy_recommendations()
        }
    
    def _get_strategy_recommendations(self) -> Dict[str, str]:
        """Generate strategy recommendations based on performance data."""
        recommendations = {}
        
        strategy_performance = self.global_stats['strategy_performance']
        
        # Find best performing strategy
        best_strategy = max(
            strategy_performance.keys(),
            key=lambda k: strategy_performance[k]['avg_success']
        )
        
        recommendations['overall_best'] = best_strategy
        
        # Recommendations for different scenarios
        recommendations['high_performance_sites'] = 'http_only'
        recommendations['javascript_heavy_sites'] = 'browser_only'
        recommendations['protected_sites'] = 'adaptive'
        recommendations['general_purpose'] = 'adaptive'
        
        return recommendations

def create_adaptive_crawler(config: Optional[Dict[str, Any]] = None) -> AdaptiveCrawler:
    """Factory function to create an AdaptiveCrawler instance."""
    return AdaptiveCrawler(config)