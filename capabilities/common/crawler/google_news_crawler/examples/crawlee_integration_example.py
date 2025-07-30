#!/usr/bin/env python3
"""
Google News Crawler + Crawlee Integration Example
================================================

Demonstrates how to use the Google News crawler with Crawlee integration 
for enhanced content downloading and analysis.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main example function demonstrating Crawlee integration."""
    
    logger.info("🚀 Starting Google News + Crawlee Integration Example")
    
    try:
        # Import Google News crawler components
        from ..api.google_news_client import create_crawlee_enhanced_gnews_client
        from ..crawlee_integration import create_crawlee_config
        
        # Import database manager (you'll need to provide a real one)
        # For this example, we'll create a placeholder
        class MockDBManager:
            async def store_articles(self, articles):
                logger.info(f"Mock storing {len(articles)} articles")
                return len(articles)
        
        db_manager = MockDBManager()
        
        # 1. Create Crawlee configuration optimized for news content
        crawlee_config = {
            'max_requests': 20,  # Limit for demo
            'max_concurrent': 3,
            'target_countries': ['ET', 'SO', 'KE'],  # Horn of Africa focus
            'enable_full_content': True,
            'min_content_length': 500,
            'enable_content_scoring': True,
            'crawl_delay': 2.0,  # Be respectful
            'max_retries': 2
        }
        
        # 2. Create enhanced Google News client with Crawlee integration
        logger.info("Creating Google News client with Crawlee integration...")
        client = await create_crawlee_enhanced_gnews_client(
            db_manager=db_manager,
            crawlee_config=crawlee_config
        )
        
        # 3. Search for news with enhanced content downloading
        logger.info("Searching for conflict-related news...")
        
        search_queries = [
            "Ethiopia conflict security",
            "Somalia humanitarian crisis", 
            "Sudan violence displacement"
        ]
        
        all_enhanced_articles = []
        
        for query in search_queries:
            logger.info(f"Searching for: {query}")
            
            # Search with Crawlee enhancement enabled
            articles = await client.search_news(
                query=query,
                countries=['ET', 'SO', 'SD', 'KE'],
                max_results=10,
                enable_crawlee=True  # Enable Crawlee content enhancement
            )
            
            all_enhanced_articles.extend(articles)
            
            # Brief pause between searches
            await asyncio.sleep(3)
        
        # 4. Analyze results
        logger.info(f"\n📊 Analysis of {len(all_enhanced_articles)} enhanced articles:")
        
        crawlee_enhanced = [a for a in all_enhanced_articles if a.get('crawlee_enhanced', False)]
        basic_articles = [a for a in all_enhanced_articles if not a.get('crawlee_enhanced', False)]
        
        logger.info(f"   ✅ Crawlee enhanced: {len(crawlee_enhanced)}")
        logger.info(f"   📰 Basic articles: {len(basic_articles)}")
        
        # Analyze content quality
        if crawlee_enhanced:
            avg_word_count = sum(a.get('word_count', 0) for a in crawlee_enhanced) / len(crawlee_enhanced)
            avg_quality_score = sum(a.get('crawlee_quality_score', 0) for a in crawlee_enhanced) / len(crawlee_enhanced)
            avg_relevance_score = sum(a.get('crawlee_relevance_score', 0) for a in crawlee_enhanced) / len(crawlee_enhanced)
            
            logger.info(f"   📈 Average word count: {avg_word_count:.0f}")
            logger.info(f"   🎯 Average quality score: {avg_quality_score:.2f}")
            logger.info(f"   🎯 Average relevance score: {avg_relevance_score:.2f}")
        
        # Show sample enhanced article
        if crawlee_enhanced:
            sample = crawlee_enhanced[0]
            logger.info(f"\n📄 Sample Enhanced Article:")
            logger.info(f"   Title: {sample.get('title', 'N/A')[:100]}...")
            logger.info(f"   URL: {sample.get('url', 'N/A')}")
            logger.info(f"   Word Count: {sample.get('word_count', 0)}")
            logger.info(f"   Quality Score: {sample.get('crawlee_quality_score', 0):.2f}")
            logger.info(f"   Extraction Method: {sample.get('content_extraction_method', 'N/A')}")
            logger.info(f"   Geographic Entities: {sample.get('geographic_entities', [])}")
            logger.info(f"   Conflict Indicators: {sample.get('conflict_indicators', [])}")
            
            if sample.get('full_content'):
                content_preview = sample['full_content'][:300]
                logger.info(f"   Content Preview: {content_preview}...")
        
        # 5. Demonstrate content extraction methods comparison
        extraction_methods = {}
        for article in crawlee_enhanced:
            method = article.get('content_extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        if extraction_methods:
            logger.info(f"\n🔧 Content Extraction Methods Used:")
            for method, count in extraction_methods.items():
                logger.info(f"   {method}: {count} articles")
        
        # 6. Cleanup
        logger.info("\n🧹 Cleaning up...")
        await client.close()
        
        logger.info("✅ Example completed successfully!")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Please ensure all dependencies are installed:")
        logger.info("pip install crawlee aiohttp asyncpg feedparser beautifulsoup4")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise

async def quick_crawlee_demo():
    """Quick demo showing basic Crawlee integration."""
    
    logger.info("🎯 Quick Crawlee Demo")
    
    try:
        from ..crawlee_integration import create_crawlee_enhancer, create_crawlee_config
        
        # Create Crawlee enhancer
        config = create_crawlee_config(
            max_requests=5,
            max_concurrent=2,
            enable_full_content=True
        )
        
        enhancer = await create_crawlee_enhancer(config)
        
        # Sample article metadata
        sample_articles = [
            {
                'title': 'Ethiopia News Article',
                'url': 'https://www.bbc.com/news/world-africa-12345',
                'description': 'Sample description',
                'published_date': datetime.now(),
                'source': 'BBC'
            }
        ]
        
        # Enhance articles
        logger.info("Enhancing sample articles...")
        enhanced = await enhancer.enhance_articles(sample_articles)
        
        logger.info(f"Enhanced {len(enhanced)} articles")
        for article in enhanced:
            logger.info(f"  - {article.title}: {article.word_count} words")
        
        await enhancer.close()
        
    except Exception as e:
        logger.error(f"Quick demo failed: {e}")

if __name__ == "__main__":
    # Run the full example
    asyncio.run(main())
    
    # Uncomment to run quick demo instead
    # asyncio.run(quick_crawlee_demo())