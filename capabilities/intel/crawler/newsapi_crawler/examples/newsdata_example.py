#!/usr/bin/env python3
"""
NewsData.io Integration Example
==============================

This example demonstrates how to use the NewsData.io client with credit management
and stealth downloading capabilities.

The NewsData.io free plan provides 200 credits (2000 articles), so this example
shows how to use them efficiently.

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the newsdata client
try:
    from ..api.newsdata_client import NewsDataClient
    from ..api.factory import create_newsdata_client
except ImportError:
    # For running as standalone script
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api.newsdata_client import NewsDataClient
    from api.factory import create_newsdata_client


async def basic_newsdata_example():
    """Basic example of using NewsData.io client."""
    logger.info("=== Basic NewsData.io Example ===")
    
    # Create client (will use NEWSDATA_API_KEY env var)
    client = create_newsdata_client()
    
    try:
        # Check initial credit status
        credit_status = client.get_credit_status()
        logger.info(f"Initial credits: {credit_status['remaining_credits']} remaining")
        logger.info(f"Can fetch approximately {credit_status['remaining_articles']} articles")
        
        # Search for Horn of Africa conflict news
        logger.info("Searching for Horn of Africa conflict news...")
        response = await client.get_latest_news(
            q="Horn of Africa conflict",
            language="en",
            size=5  # Small size to conserve credits
        )
        
        articles = response.get("results", [])
        logger.info(f"Found {len(articles)} articles")
        
        # Display some basic info about the articles
        for i, article in enumerate(articles[:3], 1):
            logger.info(f"\nArticle {i}:")
            logger.info(f"  Title: {article.get('title', 'N/A')}")
            logger.info(f"  Source: {article.get('source_name', 'N/A')}")
            logger.info(f"  Published: {article.get('pubDate', 'N/A')}")
            logger.info(f"  URL: {article.get('link', 'N/A')}")
        
        # Check credit status after request
        credit_status = client.get_credit_status()
        logger.info(f"\nCredits after search: {credit_status['remaining_credits']} remaining")
        
    finally:
        await client.close()


async def stealth_download_example():
    """Example of using NewsData.io with stealth content downloading."""
    logger.info("\n=== NewsData.io with Stealth Download Example ===")
    
    # Create client
    client = create_newsdata_client()
    
    try:
        # Check credits before starting
        credit_status = client.get_credit_status()
        logger.info(f"Starting with {credit_status['remaining_credits']} credits")
        
        if credit_status['remaining_credits'] < 2:
            logger.warning("Not enough credits for this example (need at least 2)")
            return
        
        # Search for specific conflict-related news with stealth download
        logger.info("Searching for Ethiopia conflict news with full content download...")
        
        enhanced_articles = await client.search_with_stealth_download(
            query="Ethiopia conflict violence",
            max_articles=3,  # Conservative to preserve credits
            language="en"
        )
        
        logger.info(f"Successfully processed {len(enhanced_articles)} articles")
        
        # Display enhanced articles
        for i, article in enumerate(enhanced_articles, 1):
            logger.info(f"\n--- Enhanced Article {i} ---")
            logger.info(f"Title: {article.get('title', 'N/A')}")
            logger.info(f"Source: {article.get('source_name', 'N/A')}")
            logger.info(f"Download Success: {article.get('download_success', False)}")
            
            if article.get('download_success'):
                content = article.get('extracted_text', '')
                logger.info(f"Content length: {len(content)} characters")
                # Show first 200 characters
                if content:
                    logger.info(f"Content preview: {content[:200]}...")
            else:
                logger.info(f"Download error: {article.get('download_error', 'Unknown')}")
        
        # Final credit status
        credit_status = client.get_credit_status()
        logger.info(f"\nFinal credits: {credit_status['remaining_credits']} remaining")
        
    finally:
        await client.close()


async def regional_monitoring_example():
    """Example of using NewsData.io for regional monitoring."""
    logger.info("\n=== Regional Monitoring Example ===")
    
    # Create client
    client = create_newsdata_client()
    
    try:
        credit_status = client.get_credit_status()
        if credit_status['remaining_credits'] < 3:
            logger.warning("Not enough credits for regional monitoring (need at least 3)")
            return
        
        # Define regions of interest for Horn of Africa
        regions = [
            {"query": "Somalia conflict", "country": None},
            {"query": "Ethiopia Tigray", "country": None},
            {"query": "Sudan crisis", "country": None}
        ]
        
        all_results = {}
        
        for region in regions:
            logger.info(f"Monitoring: {region['query']}")
            
            # Get articles for this region
            response = await client.get_latest_news(
                q=region["query"],
                language="en",
                size=2  # Small size to spread across regions
            )
            
            articles = response.get("results", [])
            all_results[region["query"]] = articles
            
            logger.info(f"  Found {len(articles)} articles")
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        # Summarize results
        logger.info("\n--- Regional Monitoring Summary ---")
        total_articles = 0
        for region, articles in all_results.items():
            logger.info(f"{region}: {len(articles)} articles")
            total_articles += len(articles)
        
        logger.info(f"Total articles collected: {total_articles}")
        
        # Show credit usage
        credit_status = client.get_credit_status()
        logger.info(f"Credits remaining: {credit_status['remaining_credits']}")
        
    finally:
        await client.close()


async def credit_optimization_example():
    """Example showing credit optimization strategies."""
    logger.info("\n=== Credit Optimization Example ===")
    
    client = create_newsdata_client()
    
    try:
        # Get initial status
        credit_status = client.get_credit_status()
        logger.info(f"Starting credits: {credit_status['remaining_credits']}")
        logger.info(f"Recommended page size: {credit_status['recommended_page_size']}")
        
        # Example of optimized search
        logger.info("\nPerforming optimized search...")
        
        # The client automatically optimizes page size based on remaining credits
        response = await client.get_latest_news(
            q="Horn of Africa humanitarian crisis",
            language="en"
            # Note: not specifying size - client will optimize automatically
        )
        
        articles = response.get("results", [])
        logger.info(f"Fetched {len(articles)} articles with optimized page size")
        
        # Show daily usage tracking
        credit_status = client.get_credit_status()
        today = datetime.now().strftime("%Y-%m-%d")
        daily_usage = credit_status.get('daily_usage', {})
        
        logger.info(f"\n--- Credit Usage Tracking ---")
        logger.info(f"Today's usage: {daily_usage.get(today, 0)} credits")
        logger.info(f"Remaining credits: {credit_status['remaining_credits']}")
        
        # Show optimization recommendations
        logger.info(f"\n--- Optimization Recommendations ---")
        if credit_status['remaining_credits'] > 100:
            logger.info("✓ Good credit reserves - can use normal page sizes")
        elif credit_status['remaining_credits'] > 50:
            logger.info("⚠ Moderate credits - consider smaller page sizes")
        elif credit_status['remaining_credits'] > 10:
            logger.info("⚠ Low credits - use minimal page sizes and cache aggressively")
        else:
            logger.info("❌ Very low credits - consider waiting for reset or upgrading plan")
        
    finally:
        await client.close()


async def archive_search_example():
    """Example of searching historical news archives."""
    logger.info("\n=== Archive Search Example ===")
    
    client = create_newsdata_client()
    
    try:
        credit_status = client.get_credit_status()
        if credit_status['remaining_credits'] < 2:
            logger.warning("Not enough credits for archive search")
            return
        
        # Search for historical data from last week
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Searching archives from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        response = await client.get_archive_news(
            q="Horn of Africa drought",
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
            language="en",
            size=3
        )
        
        articles = response.get("results", [])
        logger.info(f"Found {len(articles)} historical articles")
        
        # Show articles by date
        for article in articles:
            pub_date = article.get('pubDate', 'Unknown')
            title = article.get('title', 'N/A')
            logger.info(f"  {pub_date}: {title}")
        
    finally:
        await client.close()


async def main():
    """Run all examples."""
    logger.info("NewsData.io Integration Examples")
    logger.info("================================")
    
    # Check if API key is available
    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        logger.error("NEWSDATA_API_KEY environment variable not set!")
        logger.info("Please set your NewsData.io API key:")
        logger.info("export NEWSDATA_API_KEY='your_api_key_here'")
        return
    
    try:
        # Run examples
        await basic_newsdata_example()
        await stealth_download_example()
        await regional_monitoring_example()
        await credit_optimization_example()
        await archive_search_example()
        
        logger.info("\n🎉 All examples completed successfully!")
        logger.info("\nRemember to monitor your credit usage carefully with the free plan.")
        logger.info("Consider upgrading to a paid plan for higher usage volumes.")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())