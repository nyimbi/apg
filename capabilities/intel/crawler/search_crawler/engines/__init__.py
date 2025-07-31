"""
Search Engines Package
======================

Collection of search engine implementations for the search crawler.
Includes 10+ search engines for comprehensive coverage.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

from .base_search_engine import BaseSearchEngine, SearchResult, SearchResponse
from .google_search import GoogleSearchEngine
from .bing_search import BingSearchEngine
from .duckduckgo_search import DuckDuckGoSearchEngine
from .yandex_search import YandexSearchEngine
from .baidu_search import BaiduSearchEngine
from .yahoo_search import YahooSearchEngine
from .startpage_search import StartpageSearchEngine
from .searx_search import SearXSearchEngine
from .brave_search import BraveSearchEngine
from .mojeek_search import MojeekSearchEngine
from .swisscows_search import SwisscowsSearchEngine

# Engine registry for easy access
SEARCH_ENGINES = {
    'google': GoogleSearchEngine,
    'bing': BingSearchEngine,
    'duckduckgo': DuckDuckGoSearchEngine,
    'yandex': YandexSearchEngine,
    'baidu': BaiduSearchEngine,
    'yahoo': YahooSearchEngine,
    'startpage': StartpageSearchEngine,
    'searx': SearXSearchEngine,
    'brave': BraveSearchEngine,
    'mojeek': MojeekSearchEngine,
    'swisscows': SwisscowsSearchEngine
}

def get_available_engines():
    """Get list of available search engine names."""
    return list(SEARCH_ENGINES.keys())

def create_engine(engine_name: str, config=None):
    """Create a search engine instance by name."""
    if engine_name not in SEARCH_ENGINES:
        raise ValueError(f"Unknown search engine: {engine_name}")
    return SEARCH_ENGINES[engine_name](config)

__all__ = [
    'BaseSearchEngine',
    'SearchResult', 
    'SearchResponse',
    'GoogleSearchEngine',
    'BingSearchEngine',
    'DuckDuckGoSearchEngine',
    'YandexSearchEngine',
    'BaiduSearchEngine', 
    'YahooSearchEngine',
    'StartpageSearchEngine',
    'SearXSearchEngine',
    'BraveSearchEngine',
    'MojeekSearchEngine',
    'SwisscowsSearchEngine',
    'SEARCH_ENGINES',
    'get_available_engines',
    'create_engine'
]