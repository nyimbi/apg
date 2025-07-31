"""
Basic Search Crawler Demo
=========================

Simple demonstration of multi-engine search functionality.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import search crawler components
try:
    from ..core.search_crawler import SearchCrawler, SearchCrawlerConfig
    from ..engines import get_available_engines, SEARCH_ENGINES
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Search crawler components not available: {e}")
    COMPONENTS_AVAILABLE = False


async def basic_search_demo():
    """Demonstrate basic multi-engine search."""
    if not COMPONENTS_AVAILABLE:
        print("❌ Search crawler components not available!")
        return
    
    print("🔍 Basic Multi-Engine Search Demo")
    print("=" * 40)
    
    # Show available engines
    engines = get_available_engines()
    print(f"📋 Available search engines ({len(engines)}):")
    for engine in engines:
        print(f"  • {engine}")
    print()
    
    # Configure search crawler with multiple engines
    config = SearchCrawlerConfig(
        engines=['google', 'bing', 'duckduckgo', 'brave'],  # Use 4 engines
        max_results_per_engine=5,
        total_max_results=15,
        parallel_searches=True,
        download_content=False  # Skip content download for demo speed
    )
    
    crawler = SearchCrawler(config)
    
    try:
        # Perform search
        query = "Somalia conflict news"
        print(f"🔎 Searching for: '{query}'")
        print()
        
        results = await crawler.search(
            query=query,
            max_results=15
        )
        
        if results:
            print(f"✅ Found {len(results)} results from multiple engines")
            print()
            
            # Show results grouped by engine
            engine_results = {}
            for result in results:
                for engine in result.engines_found:
                    if engine not in engine_results:
                        engine_results[engine] = []
                    engine_results[engine].append(result)
            
            print("📊 Results by Engine:")
            for engine, engine_results_list in engine_results.items():
                print(f"  {engine}: {len(engine_results_list)} results")
            print()
            
            # Show top 5 results
            print("🏆 Top 5 Results:")
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   Engines: {', '.join(result.engines_found)}")
                print(f"   Relevance: {result.relevance_score:.3f}")
                if hasattr(result, 'combined_score'):
                    print(f"   Combined Score: {result.combined_score:.3f}")
                print()
            
            # Show engine statistics
            stats = crawler.get_stats()
            print("📈 Crawler Statistics:")
            print(f"  Total searches: {stats['total_searches']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Total results: {stats['total_results']}")
            print(f"  Average search time: {stats['average_search_time']:.2f}s")
            print()
            
        else:
            print("❌ No results found")
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await crawler.close()


async def engine_comparison_demo():
    """Compare results from different search engines."""
    if not COMPONENTS_AVAILABLE:
        return
    
    print("⚖️  Search Engine Comparison Demo")
    print("=" * 40)
    
    query = "Ethiopia Tigray conflict"
    engines_to_test = ['google', 'bing', 'duckduckgo', 'yandex']
    
    print(f"🔎 Comparing engines for query: '{query}'")
    print()
    
    engine_results = {}
    
    for engine in engines_to_test:
        print(f"🔍 Testing {engine}...")
        
        try:
            # Create single-engine crawler
            config = SearchCrawlerConfig(
                engines=[engine],
                max_results_per_engine=5,
                download_content=False
            )
            
            crawler = SearchCrawler(config)
            results = await crawler.search(query, max_results=5)
            
            engine_results[engine] = {
                'results': results,
                'count': len(results),
                'avg_relevance': sum(r.relevance_score for r in results) / len(results) if results else 0
            }
            
            print(f"  ✅ {len(results)} results, avg relevance: {engine_results[engine]['avg_relevance']:.3f}")
            
            await crawler.close()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            engine_results[engine] = {'results': [], 'count': 0, 'avg_relevance': 0}
    
    print()
    print("📊 Engine Comparison Summary:")
    print("-" * 30)
    
    for engine, data in engine_results.items():
        print(f"{engine:12} | {data['count']:2} results | {data['avg_relevance']:.3f} avg relevance")
    
    # Show unique vs common results
    all_urls = set()
    for data in engine_results.values():
        for result in data['results']:
            all_urls.add(result.url)
    
    print(f"\nTotal unique URLs found: {len(all_urls)}")
    
    # Find URLs found by multiple engines
    url_counts = {}
    for data in engine_results.values():
        for result in data['results']:
            url_counts[result.url] = url_counts.get(result.url, 0) + 1
    
    common_urls = [url for url, count in url_counts.items() if count > 1]
    print(f"URLs found by multiple engines: {len(common_urls)}")


async def main():
    """Run all demos."""
    await basic_search_demo()
    print("\n" + "="*50 + "\n")
    await engine_comparison_demo()


if __name__ == "__main__":
    asyncio.run(main())