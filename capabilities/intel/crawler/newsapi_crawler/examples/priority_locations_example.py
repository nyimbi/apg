#!/usr/bin/env python3
"""
Priority Locations Monitoring Example
====================================

This example demonstrates the location-optimized NewsData.io crawler specifically
targeting Aweil, Karamoja, Mandera, and Assosa for conflict monitoring.

The crawler implements a hierarchical search strategy:
1. Specific locations (Aweil, Karamoja, Mandera, Assosa)
2. District/County level (Aweil Center County, Mandera County, etc.)
3. Regional level (Northern Bahr el Ghazal, Karamoja sub-region, etc.)
4. Country level (South Sudan, Uganda, Kenya, Ethiopia)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the newsdata client and location optimizer
try:
    from ..api.newsdata_client import NewsDataClient
    from ..api.factory import create_newsdata_client
    from ..utils.location_optimizer import get_priority_locations, get_conflict_keywords
except ImportError:
    # For running as standalone script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.newsdata_client import NewsDataClient
    from api.factory import create_newsdata_client
    from utils.location_optimizer import get_priority_locations, get_conflict_keywords


async def priority_locations_comprehensive_search():
    """
    Comprehensive search across all priority locations using hierarchical strategy.
    """
    logger.info("=== Priority Locations Comprehensive Search ===")
    
    client = create_newsdata_client()
    
    try:
        # Show priority locations
        locations = get_priority_locations()
        logger.info("Priority locations configured:")
        for key, location in locations.items():
            logger.info(f"  {key.upper()}: {location.specific_location} -> {location.district} -> {location.region} -> {location.country}")
        
        # Check initial credits
        credit_status = client.get_credit_status()
        logger.info(f"\nStarting with {credit_status['remaining_credits']} credits")
        
        if credit_status['remaining_credits'] < 15:
            logger.warning("Limited credits available. Consider using smaller credit budget.")
            max_credits = min(10, credit_status['remaining_credits'] - 2)
        else:
            max_credits = 15
        
        # Perform comprehensive search
        logger.info(f"\nStarting comprehensive search with {max_credits} credits...")
        results = await client.search_priority_locations(
            locations=["aweil", "karamoja", "mandera", "assosa"],
            max_credits=max_credits,
            include_stealth_download=True
        )
        
        # Display results summary
        summary = results.get("_summary", {})
        logger.info(f"\n=== SEARCH RESULTS SUMMARY ===")
        logger.info(f"Total articles found: {summary.get('total_articles', 0)}")
        logger.info(f"Credits used: {summary.get('total_credits_used', 0)}")
        logger.info(f"Credits remaining: {summary.get('credits_remaining', 0)}")
        
        # Show detailed results by location
        for location, data in results.items():
            if location.startswith("_"):  # Skip summary
                continue
                
            logger.info(f"\n--- {location.upper()} RESULTS ---")
            logger.info(f"Articles found: {data.get('total_articles_found', 0)}")
            logger.info(f"Final search level: {data.get('final_level', 'N/A')}")
            logger.info(f"Credits used: {data.get('credits_used', 0)}")
            logger.info(f"Search levels tried: {', '.join(data.get('search_levels_tried', []))}")
            
            # Show top articles
            articles = data.get('articles', [])
            for i, article in enumerate(articles[:3], 1):
                logger.info(f"\n  Article {i}:")
                logger.info(f"    Title: {article.get('title', 'N/A')}")
                logger.info(f"    Source: {article.get('source_name', 'N/A')}")
                logger.info(f"    Published: {article.get('pubDate', 'N/A')}")
                logger.info(f"    Relevance Score: {article.get('relevance_score', 0):.1f}")
                
                if article.get('download_success'):
                    content_len = len(article.get('extracted_text', ''))
                    logger.info(f"    Full Content: {content_len} characters extracted")
                    
                    # Show content preview
                    content = article.get('extracted_text', '')
                    if content:
                        preview = content[:200].replace('\n', ' ')
                        logger.info(f"    Preview: {preview}...")
                else:
                    logger.info(f"    Content download: Failed")
        
    finally:
        await client.close()


async def single_location_monitoring():
    """
    Demonstrate focused monitoring of a single priority location.
    """
    logger.info("\n=== Single Location Monitoring Example ===")
    
    client = create_newsdata_client()
    
    try:
        # Monitor Aweil for last 24 hours
        logger.info("Monitoring AWEIL for conflict activity in last 24 hours...")
        
        results = await client.monitor_location(
            location="aweil",
            hours_back=24,
            max_credits=3
        )
        
        logger.info(f"Monitoring Results:")
        logger.info(f"  Location: {results['location'].upper()}")
        logger.info(f"  Period: {results['monitoring_period_hours']} hours")
        logger.info(f"  Articles found: {results['articles_found']}")
        logger.info(f"  Credits used: {results['credits_used']}")
        
        # Show found articles
        for i, article in enumerate(results['articles'][:5], 1):
            logger.info(f"\n  Recent Article {i}:")
            logger.info(f"    Title: {article.get('title', 'N/A')}")
            logger.info(f"    Published: {article.get('pubDate', 'N/A')}")
            logger.info(f"    URL: {article.get('link', 'N/A')}")
        
    finally:
        await client.close()


async def alert_scanning_example():
    """
    Demonstrate rapid alert scanning for critical incidents.
    """
    logger.info("\n=== Alert Scanning Example ===")
    
    client = create_newsdata_client()
    
    try:
        # Perform rapid alert scan
        logger.info("Performing rapid alert scan across all priority locations...")
        
        alert_results = await client.alert_scan(
            alert_keywords=["killed", "attack", "violence", "raid", "explosion"],
            max_credits=5
        )
        
        logger.info(f"\n=== ALERT SCAN RESULTS ===")
        logger.info(f"Total alerts: {alert_results['alert_count']}")
        logger.info(f"High urgency alerts: {alert_results['high_urgency_count']}")
        logger.info(f"Credits used: {alert_results['credits_used']}")
        logger.info(f"Locations scanned: {alert_results['locations_scanned']}")
        
        # Show alerts if any
        alerts = alert_results.get('alerts', [])
        if alerts:
            logger.warning(f"\n⚠️  {len(alerts)} ALERTS DETECTED:")
            
            for i, alert in enumerate(alerts[:5], 1):
                urgency_icon = "🔴" if alert['urgency'] == 'high' else "🟡"
                hours_ago = alert.get('hours_ago', 'unknown')
                if isinstance(hours_ago, (int, float)):
                    time_str = f"{hours_ago:.1f} hours ago"
                else:
                    time_str = "recent"
                
                logger.warning(f"  {urgency_icon} Alert {i} [{alert['urgency'].upper()}]:")
                logger.warning(f"    Location: {alert['location'].upper()}")
                logger.warning(f"    Keyword: {alert['keyword']}")
                logger.warning(f"    Time: {time_str}")
                logger.warning(f"    Title: {alert['article'].get('title', 'N/A')}")
        else:
            logger.info("✅ No urgent alerts detected")
        
    finally:
        await client.close()


async def adaptive_credit_management_example():
    """
    Demonstrate adaptive credit management for long-term monitoring.
    """
    logger.info("\n=== Adaptive Credit Management Example ===")
    
    client = create_newsdata_client()
    
    try:
        credit_status = client.get_credit_status()
        initial_credits = credit_status['remaining_credits']
        
        logger.info(f"Starting credits: {initial_credits}")
        
        # Adaptive strategy based on available credits
        if initial_credits > 50:
            # High credit mode: comprehensive monitoring
            logger.info("HIGH CREDIT MODE: Comprehensive monitoring")
            locations = ["aweil", "karamoja", "mandera", "assosa"]
            max_credits = 20
            include_download = True
            
        elif initial_credits > 20:
            # Medium credit mode: focused monitoring
            logger.info("MEDIUM CREDIT MODE: Focused monitoring")
            locations = ["aweil", "mandera"]  # Focus on 2 locations
            max_credits = 10
            include_download = True
            
        elif initial_credits > 5:
            # Low credit mode: alert scanning only
            logger.info("LOW CREDIT MODE: Alert scanning only")
            alert_results = await client.alert_scan(max_credits=5)
            logger.info(f"Alert scan completed: {alert_results['alert_count']} alerts found")
            return
            
        else:
            # Emergency mode: minimal monitoring
            logger.warning("EMERGENCY MODE: Minimal monitoring")
            locations = ["aweil"]  # Single location
            max_credits = 2
            include_download = False
        
        # Execute the adaptive strategy
        results = await client.search_priority_locations(
            locations=locations,
            max_credits=max_credits,
            include_stealth_download=include_download
        )
        
        summary = results.get("_summary", {})
        final_credits = summary.get('credits_remaining', 0)
        
        logger.info(f"\nAdaptive monitoring completed:")
        logger.info(f"  Articles found: {summary.get('total_articles', 0)}")
        logger.info(f"  Credits used: {initial_credits - final_credits}")
        logger.info(f"  Credits remaining: {final_credits}")
        logger.info(f"  Efficiency: {summary.get('total_articles', 0) / max(1, initial_credits - final_credits):.1f} articles/credit")
        
    finally:
        await client.close()


async def conflict_keyword_analysis():
    """
    Analyze which conflict keywords are most effective for each location.
    """
    logger.info("\n=== Conflict Keyword Analysis ===")
    
    client = create_newsdata_client()
    
    try:
        # Get available conflict keywords
        keywords = get_conflict_keywords()
        logger.info(f"Available conflict keywords: {len(keywords)}")
        logger.info(f"High-priority keywords: {keywords[:10]}")
        
        # Test keyword effectiveness for Aweil
        location = "aweil"
        test_keywords = ["killed", "violence", "attack", "raid", "conflict"]
        
        logger.info(f"\nTesting keyword effectiveness for {location.upper()}:")
        
        keyword_results = {}
        for keyword in test_keywords:
            try:
                query = f'"Aweil" {keyword}'
                response = await client.get_latest_news(
                    q=query,
                    language="en",
                    size=5
                )
                
                articles = response.get("results", [])
                relevant_count = len([a for a in articles if keyword.lower() in a.get('title', '').lower()])
                
                keyword_results[keyword] = {
                    "total_articles": len(articles),
                    "relevant_articles": relevant_count,
                    "effectiveness": relevant_count / max(1, len(articles))
                }
                
                logger.info(f"  {keyword}: {len(articles)} articles, {relevant_count} relevant (effectiveness: {keyword_results[keyword]['effectiveness']:.1%})")
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error testing keyword '{keyword}': {str(e)}")
                keyword_results[keyword] = {"error": str(e)}
        
        # Recommend best keywords
        effective_keywords = [(k, v['effectiveness']) for k, v in keyword_results.items() 
                            if 'effectiveness' in v and v['effectiveness'] > 0.3]
        effective_keywords.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"\nMost effective keywords for {location.upper()}:")
        for keyword, effectiveness in effective_keywords[:3]:
            logger.info(f"  {keyword}: {effectiveness:.1%} effectiveness")
        
    finally:
        await client.close()


async def main():
    """Run all priority location examples."""
    logger.info("Priority Locations NewsData.io Examples")
    logger.info("========================================")
    
    # Check if API key is available
    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        logger.error("NEWSDATA_API_KEY environment variable not set!")
        logger.info("Please set your NewsData.io API key:")
        logger.info("export NEWSDATA_API_KEY='your_api_key_here'")
        return
    
    # Show priority locations info
    locations = get_priority_locations()
    logger.info(f"\nConfigured for {len(locations)} priority locations:")
    for key, location in locations.items():
        logger.info(f"  • {location.specific_location} ({location.country})")
    
    try:
        # Run examples in order of increasing complexity
        await single_location_monitoring()
        await alert_scanning_example()
        await adaptive_credit_management_example()
        await conflict_keyword_analysis()
        await priority_locations_comprehensive_search()
        
        logger.info("\n🎉 All priority location examples completed successfully!")
        logger.info("\n📍 Key Features Demonstrated:")
        logger.info("  ✓ Hierarchical location search (specific → district → region → country)")
        logger.info("  ✓ Conflict-focused keyword targeting")
        logger.info("  ✓ Credit-optimized search strategies")
        logger.info("  ✓ Real-time alert scanning")
        logger.info("  ✓ Adaptive monitoring based on credit availability")
        logger.info("  ✓ Full content extraction with stealth crawler")
        
        logger.info("\n⚠️  Important Notes:")
        logger.info("  • Monitor your credit usage carefully")
        logger.info("  • Use alert_scan() for rapid incident detection")
        logger.info("  • Use monitor_location() for focused monitoring")
        logger.info("  • Use search_priority_locations() for comprehensive analysis")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())