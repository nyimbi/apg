"""
APG Crawler Capability - Unified Adapter System
===============================================

Adapters for integrating existing specialized crawlers:
- SearchCrawler: Multi-engine search orchestration  
- GdeltCrawler: Global events and media monitoring
- GoogleNewsCrawler: News aggregation and processing
- TwitterCrawler: Social media intelligence
- YouTubeCrawler: Video content analysis

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from .base_adapter import BaseSpecializedCrawlerAdapter, AdapterResult
from .search_adapter import SearchCrawlerAdapter
from .gdelt_adapter import GdeltCrawlerAdapter
from .google_news_adapter import GoogleNewsCrawlerAdapter
from .twitter_adapter import TwitterCrawlerAdapter
from .youtube_adapter import YouTubeCrawlerAdapter
from .unified_adapter_manager import UnifiedAdapterManager

__all__ = [
	'BaseSpecializedCrawlerAdapter',
	'AdapterResult',
	'SearchCrawlerAdapter',
	'GdeltCrawlerAdapter', 
	'GoogleNewsCrawlerAdapter',
	'TwitterCrawlerAdapter',
	'YouTubeCrawlerAdapter',
	'UnifiedAdapterManager'
]