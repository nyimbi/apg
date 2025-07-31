#!/usr/bin/env python3
"""
Target Locations Conflict Search Demo
=====================================

Demonstrates the enhanced search crawler with stealth capabilities
for monitoring conflicts in specific target locations.

Features demonstrated:
- Hierarchical location search (specific -> district -> country)
- Stealth article downloading via news_crawler integration
- 10 search engines for comprehensive coverage
- Conflict keyword optimization
- Minimum 50 articles per location target

Author: Lindela Development Team
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from ..core.conflict_search_crawler import (
        create_target_location_crawler,
        ConflictSearchResult
    )
    CRAWLER_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import crawler: {e}")
    CRAWLER_AVAILABLE = False


async def demo_target_location_search():
    """Demonstrate target location conflict search."""
    if not CRAWLER_AVAILABLE:
        print("❌ Crawler not available - check imports")
        return
    
    print("🚀 Target Locations Conflict Search Demo")
    print("=" * 50)
    
    # Create optimized crawler for target locations
    print("📡 Initializing conflict search crawler...")
    crawler = create_target_location_crawler(
        target_locations=['Aweil', 'Karamoja', 'Mandera', 'Assosa'],
        min_articles_per_location=50,
        enable_stealth=True,
        stealth_aggressive=False
    )
    
    try:
        # Test single location hierarchical search
        print("\n🎯 Testing hierarchical search for Aweil...")
        start_time = time.time()
        
        aweil_results = await crawler.search_target_locations(
            target_location='Aweil',
            max_results=20,  # For demo, use smaller number
            time_range='week'
        )
        
        search_time = time.time() - start_time
        print(f"⏱️  Search completed in {search_time:.2f} seconds")
        print(f"📊 Found {len(aweil_results)} results for Aweil")
        
        # Display top results
        print("\n📰 Top Aweil Results:")
        for i, result in enumerate(aweil_results[:5], 1):
            print(f"{i}. {result.title[:80]}...")
            print(f"   🔗 {result.url}")
            print(f"   📈 Conflict Score: {result.conflict_score:.2f}")
            print(f"   🛡️ Stealth Content: {'✅' if result.content and len(result.content) > 500 else '❌'}")
            if result.locations_mentioned:
                locations = [loc['name'] for loc in result.locations_mentioned[:3]]
                print(f"   📍 Locations: {', '.join(locations)}")
            if result.escalation_indicators:
                print(f"   ⚠️  Escalation: {', '.join(result.escalation_indicators)}")
            print()
        
        # Test all target locations
        print("\n🌍 Testing all target locations...")
        start_time = time.time()
        
        all_results = await crawler.search_target_locations(
            max_results=15,  # For demo
            time_range='week'
        )
        
        search_time = time.time() - start_time
        print(f"⏱️  All locations search completed in {search_time:.2f} seconds")
        
        # Analyze results by location
        location_stats = analyze_results_by_location(all_results)
        print("\n📊 Results Summary by Location:")
        for location, stats in location_stats.items():
            print(f"📍 {location}:")
            print(f"   📄 Articles: {stats['count']}")
            print(f"   📈 Avg Score: {stats['avg_score']:.2f}")
            print(f"   🛡️ Stealth Success: {stats['stealth_success']}/{stats['count']}")
            print(f"   ⚠️  Alerts: {stats['alerts']}")
        
        # Test search engine coverage
        print("\n🔍 Search Engine Coverage:")
        engine_stats = analyze_engine_coverage(all_results)
        for engine, count in engine_stats.items():
            print(f"   {engine}: {count} results")
        
        # Show crawler statistics
        print("\n📊 Crawler Statistics:")
        stats = crawler.get_conflict_stats()
        print(f"   🔄 Total Searches: {stats.get('total_searches', 0)}")
        print(f"   ✅ Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"   📄 Total Results: {stats.get('total_results', 0)}")
        print(f"   🚨 Total Alerts: {stats.get('total_alerts', 0)}")
        print(f"   📊 Tracked Conflicts: {stats.get('tracked_conflicts', 0)}")
        
        # Show conflict analysis
        print("\n🔬 Conflict Analysis:")
        conflict_types = {}
        high_conflict_count = 0
        
        for result in all_results:
            if result.conflict_type:
                conflict_types[result.conflict_type] = conflict_types.get(result.conflict_type, 0) + 1
            if result.conflict_score >= 0.7:
                high_conflict_count += 1
        
        print(f"   🔥 High Conflict Articles (≥0.7): {high_conflict_count}")
        print("   📋 Conflict Types:")
        for conflict_type, count in conflict_types.items():
            print(f"      {conflict_type}: {count}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logging.exception("Demo error details:")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await crawler.cleanup()
        print("✅ Demo completed")


def analyze_results_by_location(results: List[ConflictSearchResult]) -> Dict[str, Dict[str, Any]]:
    """Analyze results grouped by detected location."""
    location_stats = {}
    
    for result in results:
        # Try to determine primary location from mentions
        primary_location = "Unknown"
        if result.locations_mentioned:
            primary_location = result.locations_mentioned[0]['name']
        elif result.primary_location:
            primary_location = result.primary_location['name']
        
        # Check if it matches any target location
        target_locations = ['Aweil', 'Karamoja', 'Mandera', 'Assosa']
        for target in target_locations:
            if target.lower() in result.title.lower() or target.lower() in result.snippet.lower():
                primary_location = target
                break
        
        if primary_location not in location_stats:
            location_stats[primary_location] = {
                'count': 0,
                'total_score': 0.0,
                'stealth_success': 0,
                'alerts': 0
            }
        
        stats = location_stats[primary_location]
        stats['count'] += 1
        stats['total_score'] += result.conflict_score
        
        if result.content and len(result.content) > 500:
            stats['stealth_success'] += 1
        
        if result.requires_alert:
            stats['alerts'] += 1
    
    # Calculate averages
    for location, stats in location_stats.items():
        if stats['count'] > 0:
            stats['avg_score'] = stats['total_score'] / stats['count']
        else:
            stats['avg_score'] = 0.0
    
    return location_stats


def analyze_engine_coverage(results: List[ConflictSearchResult]) -> Dict[str, int]:
    """Analyze which search engines provided results."""
    engine_stats = {}
    
    for result in results:
        for engine in result.engines_found:
            engine_stats[engine] = engine_stats.get(engine, 0) + 1
    
    return dict(sorted(engine_stats.items(), key=lambda x: x[1], reverse=True))


async def demo_single_location_deep_search():
    """Demonstrate deep hierarchical search for a single location."""
    print("\n🔍 Deep Hierarchical Search Demo - Mandera")
    print("=" * 50)
    
    if not CRAWLER_AVAILABLE:
        print("❌ Crawler not available")
        return
    
    crawler = create_target_location_crawler(
        target_locations=['Mandera'],
        min_articles_per_location=75,  # Higher target for deep search
        enable_stealth=True,
        stealth_aggressive=True  # More aggressive for challenging sites
    )
    
    try:
        print("🎯 Performing deep search for Mandera conflicts...")
        start_time = time.time()
        
        results = await crawler.search_target_locations(
            target_location='Mandera',
            max_results=75,
            time_range='month'  # Broader time range
        )
        
        search_time = time.time() - start_time
        print(f"⏱️  Deep search completed in {search_time:.2f} seconds")
        print(f"📊 Found {len(results)} results for Mandera")
        
        # Analyze content quality
        stealth_success = sum(1 for r in results if r.content and len(r.content) > 500)
        high_conflict = sum(1 for r in results if r.conflict_score >= 0.6)
        with_locations = sum(1 for r in results if r.locations_mentioned)
        
        print(f"\n📈 Quality Analysis:")
        print(f"   🛡️ Stealth Downloads: {stealth_success}/{len(results)} ({stealth_success/len(results)*100:.1f}%)")
        print(f"   🔥 High Conflict (≥0.6): {high_conflict}/{len(results)} ({high_conflict/len(results)*100:.1f}%)")
        print(f"   📍 With Locations: {with_locations}/{len(results)} ({with_locations/len(results)*100:.1f}%)")
        
        # Show most relevant results
        print(f"\n🏆 Top 5 Most Relevant Results:")
        top_results = sorted(results, key=lambda x: x.conflict_score, reverse=True)[:5]
        
        for i, result in enumerate(top_results, 1):
            print(f"{i}. Score: {result.conflict_score:.3f} | {result.title[:60]}...")
            print(f"   🔗 {result.url}")
            if result.escalation_indicators:
                print(f"   ⚠️  Escalation: {', '.join(result.escalation_indicators[:3])}")
            if result.conflict_type:
                print(f"   📋 Type: {result.conflict_type}")
            print()
    
    except Exception as e:
        print(f"❌ Deep search failed: {e}")
        logging.exception("Deep search error details:")
    
    finally:
        await crawler.cleanup()


if __name__ == "__main__":
    print("🌍 Lindela Target Locations Conflict Monitoring")
    print("🎯 Focusing on: Aweil, Karamoja, Mandera, Assosa")
    print("🛡️ Using stealth news crawler integration")
    print("🔍 Hierarchical search: specific → district → country")
    print("🌐 10 search engines for comprehensive coverage")
    print()
    
    # Run demos
    asyncio.run(demo_target_location_search())
    print("\n" + "="*60)
    asyncio.run(demo_single_location_deep_search())